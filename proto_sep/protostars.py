from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from astropy.coordinates import SkyCoord

from .utils.logging import setup_logger

log = setup_logger(__name__)


@dataclass
class Protostar:
    name: str
    location: SkyCoord
    inclination: float
    inclination_error: float
    rmaj: float
    tbol: float


@dataclass
class Group:

    protostars: List[Protostar]
    separation: np.ndarray = field(init=False)
    inclination_difference: np.ndarray = field(init=False)
    inclination_difference_error: np.ndarray = field(init=False)
    rmaj: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        (
            self.separation,
            self.inclination_difference,
            self.inclination_difference_error,
            self.rmaj,
        ) = self._compute_separations()

    def _compute_separations(self) -> Tuple[np.ndarray]:

        seps = []
        inc_diff = []
        inc_diff_err = []
        rmaj = []

        for ps in self.protostars[1:]:

            seps.append(self.protostars[0].location.separation(ps.location).rad)

            inc_diff.append(
                np.abs(self.protostars[0].inclination - ps.inclination)
            )

            inc_diff_err.append(
                np.sqrt(
                    self.protostars[0].inclination_error ** 2
                    + ps.inclination_error**2
                )
            )

            rmaj.append(max([self.protostars[0].rmaj, ps.rmaj]))

            return (
                np.array(seps),
                np.array(inc_diff),
                np.array(inc_diff_err),
                np.array(rmaj),
            )


@dataclass
class Region:
    groups: List[Group]

    @classmethod
    def from_file(cls, file_name: str) -> "Region":

        data = pd.read_csv(file_name, index_col="Name")

        names = []
        fields = []
        objects = []

        for k, v in data.iterrows():

            name, field, obj = k.split("_")

            fields.append(int(field))
            objects.append(int(obj))
            names.append(k)

        fields = np.array(fields)
        objects = np.array(objects)

        names = np.array(names)

        group_names = []

        for f in np.unique(fields):

            idx = fields == f

            if len(objects[idx]) > 1:

                group_names.append(names[idx])

        groups = []
        for group in group_names:

            tmp = []

            for name in group:

                if np.isfinite(data.loc[name].RA):

                    tmp.append(
                        Protostar(
                            name=name,
                            location=SkyCoord(
                                data.loc[name].RA,
                                data.loc[name].DEC,
                                frame="icrs",
                                unit="deg",
                            ),
                            inclination=data.loc[name].Inc,
                            inclination_error=data.loc[name].Inc_err,
                            rmaj=data.loc[name].Rmaj,
                            tbol=data.loc[name].Tbol0,
                        )
                    )

            if len(tmp) > 1:

                groups.append(Group(tmp))

            else:

                log.warning(f"{name.split('_')[0]} must have contained NaNs")

        return cls(groups)


@dataclass
class Catalog:
    regions: List[Region]

    @classmethod
    def from_files(cls, *files: List[str]) -> "Catalog":

        regions = [Region.from_file(f) for f in files]

        return cls(regions)

    def _walk_regions_and_groups(self, variable: str) -> np.ndarray:

        output = []

        for region in self.regions:

            for group in region.groups:

                output.extend(asdict(group)[variable])

        return np.array(output)

    def _walk_protostars(self, variable: str) -> np.ndarray:

        output = []

        for region in self.regions:

            for group in region.groups:

                for protostar in group.protostars:

                    output.extend(asdict(protostar)[variable])

        return np.array(output)

    @property
    def separation(self) -> np.ndarray:
        return self._walk_regions_and_groups("separation")

    @property
    def inclination_difference(self) -> np.ndarray:
        return self._walk_regions_and_groups("inclination_difference")

    @property
    def inclination_difference_error(self) -> np.ndarray:
        return self._walk_regions_and_groups("inclination_difference_error")

    @property
    def rmaj(self) -> np.ndarray:
        return self._walk_regions_and_groups("rmaj")

    @property
    def all_rmaj(self) -> np.ndarray:
        return self._walk_protostars("rmaj")

    @property
    def all_tbol(self) -> np.ndarray:
        return self._walk_protostars("tbol")
