import glob
import os

import bioio
import numpy as np

from pint import UndefinedUnitError, UnitRegistry

from ashlar.reg import warn_data, Reader

ureg = UnitRegistry()


def _get_ome(img):
    try:
        return img.ome_metadata
    except NotImplementedError:
        return None


class BioImageIOMetadata:

    def __init__(self, paths):
        self._images = []

        for path in paths:
            _, ext = os.path.splitext(path)
            image_reader = None
            self._images.append(bioio.BioImage(path, reconstruct_mosaic=False, reader=image_reader))

    @property
    def _num_images(self):
        ome_metadata = _get_ome(self._images[0])
        if ome_metadata is None:
            return len(self._images)

        return len(self._images) if len(self._images) > 1 else len(ome_metadata.images)

    @property
    def pixel_dtype(self):
        return self._images[0].dask_data.dtype

    @property
    def num_channels(self):
        ome_metadata = _get_ome(self._images[0])
        if ome_metadata is None:
            return self._images[0].xarray_dask_data.sizes["C"]
        else:
            return len(ome_metadata.images[0].pixels.channels)

    @property
    def pixel_size(self):
        ome_metadata = _get_ome(self._images[0])
        if ome_metadata is not None:
            values = [
                ome_metadata.images[0].pixels.physical_size_y,
                ome_metadata.images[0].pixels.physical_size_x,
            ]
            physical_size_y_unit = ome_metadata.images[0].pixels.physical_size_y_unit.value
            physical_size_x_unit = ome_metadata.images[0].pixels.physical_size_x_unit.value
        else:
            img = self._images[0]
            metadata = img.metadata["multiscales"][0]["metadata"]
            values = [metadata["physical_size_y"], metadata["physical_size_x"]]
            physical_size_y_unit = metadata["physical_size_y_unit"]
            physical_size_x_unit = metadata["physical_size_x_unit"]

        if physical_size_y_unit is not None and physical_size_x_unit is not None:
            try:
                values[0] = (
                        values[0]
                        * ureg.parse_expression(physical_size_y_unit).to("micrometer").magnitude
                )
                values[1] = (
                        values[1]
                        * ureg.parse_expression(physical_size_x_unit).to("micrometer").magnitude
                )
            except UndefinedUnitError:
                warn_data("Unknown physical size units. Assuming µm")
        else:
            warn_data("Unknown physical size units. Assuming µm")
        if not np.isclose(values[0], values[1], rtol=1e-4, atol=0):
            raise Exception(f"Can't handle non-square pixels {values[::-1]}")
        if values[0] != values[1]:
            warn_data(
                f"Pixel size is slightly non-square {values[::-1]}. Using"
                f" {values[1]} for both dimensions."
            )
        return values[1]

    def tile_position(self, i):
        image_index = 0 if len(self._images) > 1 else i
        if len(self._images) > 1:
            img = self._images[i]
        else:
            img = self._images[0]
            img.set_scene(i)
        ome_metadata = _get_ome(img)
        if ome_metadata is not None:
            values = [
                ome_metadata.images[image_index].pixels.planes[0].position_y,
                ome_metadata.images[image_index].pixels.planes[0].position_x,
            ]
            physical_size_y_unit = (
                ome_metadata.images[image_index].pixels.planes[0].position_y_unit.value
            )
            physical_size_x_unit = (
                ome_metadata.images[image_index].pixels.planes[0].position_x_unit.value
            )
        else:
            metadata = img.metadata["multiscales"][0]["metadata"]
            values = [metadata["position_y"], metadata["position_x"]]
            physical_size_y_unit = metadata["position_y_unit"]
            physical_size_x_unit = metadata["position_x_unit"]
        if physical_size_y_unit is not None and physical_size_x_unit is not None:
            try:
                values[0] = (
                        values[0]
                        * ureg.parse_expression(physical_size_y_unit).to("micrometer").magnitude
                )
                values[1] = (
                        values[1]
                        * ureg.parse_expression(physical_size_x_unit).to("micrometer").magnitude
                )
            except UndefinedUnitError:
                warn_data("Unknown stage coordinate size units. Assuming µm")
        else:
            warn_data("Unknown stage coordinate size units. Assuming µm")
        position_microns = np.array(values, dtype=float)
        # Invert Y so that stage position coordinates and image pixel
        # coordinates are aligned (most formats seem to work this way).
        position_microns *= [-1, 1]
        position_pixels = position_microns / self.pixel_size
        return position_pixels

    def tile_size(self, i):
        if len(self._images) > 1:
            img = self._images[i]
        else:
            img = self._images[0]
            img.set_scene(i)
        ome_metadata = _get_ome(img)
        if ome_metadata is not None:
            values = [
                ome_metadata.images[0].pixels.size_y,
                ome_metadata.images[0].pixels.size_x,
            ]
        else:
            img = self._images[0]
            values = img.shape[-2:]

        return np.array(values, dtype=int)

    @property
    def grid_dimensions(self):
        pos = self.positions
        shape = np.array([len(set(pos[:, d])) for d in range(2)])
        if np.prod(shape) != self.num_images:
            raise ValueError("Series positions do not form a grid")
        return shape

    @property
    def num_images(self):
        return self._num_images

    @property
    def positions(self):
        if not hasattr(self, "_positions"):
            self._positions = np.vstack([self.tile_position(i) for i in range(self._num_images)])
        return self._positions

    @property
    def size(self):
        if not hasattr(self, "_size"):
            s0 = self.tile_size(0)
            image_ids = range(1, self._num_images)
            if any(any(self.tile_size(i) != s0) for i in image_ids):
                raise ValueError("Image series must all have the same dimensions")
            self._size = s0
        return self._size

    @property
    def centers(self):
        return self.positions + self.size / 2

    @property
    def origin(self):
        return self.positions.min(axis=0)

    def read(self, series, c, z):

        if len(self._images) > 1:
            img = self._images[series]
        else:
            img = self._images[0]
            img.set_scene(int(series))
        if z == "max":
            size_z = img.dask_data.shape[2]
            if size_z > 1:
                array = []
                for z in range(size_z):
                    array.append(img.dask_data[:, c, z].squeeze().compute())
                return np.array(array).max(axis=0)
            return img.dask_data[:, c, 0].squeeze().compute()
        else:
            return img.dask_data[:, c, z].squeeze().compute()

    @property
    def num_plates(self):
        raise ValueError()

    @property
    def num_wells(self):
        raise ValueError()

    @property
    def plate_well_series(self):
        raise ValueError()

    def plate_name(self, i):
        raise ValueError()

    @property
    def well_naming(self):
        raise ValueError()


class BioIOReader(Reader):
    def __init__(self, path, **kwargs):
        self.paths = glob.glob(path)
        assert len(self.paths) > 0, "No paths found"
        self.z_index = kwargs.get("z", "max")
        self.metadata = BioImageIOMetadata(self.paths)

    def __del__(self):
        for path in self.paths:
            if path is not None:
                os.remove(path)

    def get_tile(self, i):
        index = i if len(self.paths) > 1 else 0
        return self.paths[index]

    def read(self, series, c):
        return self.metadata.read(series, c, self.z_index)
