import json
import re
import itertools
import warnings
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from pathlib import Path
from pprint import pprint
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.mast import Observations, Catalogs
from astropy.coordinates import SkyCoord, Distance, Galactocentric
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.table import Table
from astroquery.vizier import Vizier
from loguru import logger

current_dir = Path(__file__).parent
DATA_PATH = current_dir.parent / "data"
    
class Target:
    def __init__(self, target_name=None,
                 ra_deg=None, dec_deg=None, gaiaDR2id=None, verbose=True):
        self.target_name = target_name
        self.verbose = verbose
        self.tfop_info = None
        if (ra_deg is not None) and (dec_deg is not None):
            self.target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        else:
            self.tfop_info = self.query_tfop_info()
            self.parse_tfop_info()
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.vizier_tables = None
        self.search_radius = 1 * u.arcmin
        self.gaia_sources = None
        self.target_names = None

    def query_vizier(self, radius=3, verbose=None):
        """
        Useful to get relevant catalogs from literature
        See:
        https://astroquery.readthedocs.io/en/latest/vizier/vizier.html
        """
        verbose = self.verbose if verbose is None else verbose
        radius = self.search_radius if radius is None else radius * u.arcsec
        if verbose:
            print(
                f"Searching Vizier: ({self.target_coord.to_string()}) with radius={radius}."
            )
        # standard column sorted in increasing distance
        v = Vizier(
            columns=["*", "+_r"],
            # column_filters={"Vmag":">10"},
            # keywords=['stars:white_dwarf']
        )
        if self.vizier_tables is None:
            tables = v.query_region(self.target_coord, radius=radius)
            if tables is None:
                print("No result from Vizier.")
            else:
                if verbose:
                    print(f"{len(tables)} tables found.")
                    pprint(
                        {
                            k: tables[k]._meta["description"]
                            for k in tables.keys()
                        }
                    )
                self.vizier_tables = tables
        else:
            tables = self.vizier_tables
        return tables
    
    def query_tfop_info(self) -> dict:
        if self.tfop_info:
            return self.tfop_info
        else:
            base_url = "https://exofop.ipac.caltech.edu/tess"
            url = f"{base_url}/target.php?id={self.target_name.replace(' ','')}&json"
            response = urlopen(url)
            assert response.code == 200, "Failed to get data from ExoFOP-TESS"
            try:
                data_json = json.loads(response.read())
                return data_json
            except Exception:
                raise ValueError(f"No TIC data found for {self.target_name}")
                
    def parse_tfop_info(self) -> None:
        """
        Parse the TFOP info to get the star names, Gaia name, Gaia ID, and target coordinates.
        """
        self.star_names = np.array(
            self.tfop_info.get("basic_info")["star_names"].split(", ")
        )
        if self.target_name is None:
            self.target_name = self.star_names[0]
            
        if self.verbose:
            logger.info(f"Catalog names:")
            for n in self.star_names:
                print(f"\t{n}")
        self.gaia_name = self.star_names[
            np.array([i[:4].lower() == "gaia" for i in self.star_names])
        ][0]
        self.gaiaid = int(self.gaia_name.split()[-1])
        ra, dec = (
            self.tfop_info.get("coordinates")["ra"],
            self.tfop_info.get("coordinates")["dec"],
        )
        self.target_coord = SkyCoord(ra=ra, dec=dec, unit="degree")

        if self.target_name.lower()[:3] == "toi":
            parts = self.target_name.split("-")
            if len(parts)>1:
                self.toiid = parts[-1]
            else:
                self.toiid = int(float(self.target_name.replace(" ","")[3:]))
        else:
            idx = [i[:3].lower() == "toi" for i in self.star_names]
            if sum(idx) > 0:
                toiid = int(self.star_names[idx][0].split("-")[-1])
            else:
                toiid = None
            self.toiid = toiid
        self.ticid = int(self.tfop_info.get("basic_info")["tic_id"])
        if self.ticid is not None:
            self.query_name = f"TIC{self.ticid}"
        else:
            self.query_name = self.target_name.replace("-", " ")

    def get_params_from_tfop(self, name="planet_parameters", idx=None) -> dict:
        if self.tfop_info is None:
            self.query_tfop_info()
        params_dict = self.tfop_info.get(name)
        if idx is None:
            key = "pdate" if name == "planet_parameters" else "sdate"
            # get the latest parameter based on upload date
            dates = []
            for d in params_dict:
                t = d.get(key)
                dates.append(t)
            df = pd.DataFrame({"date": dates})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            idx = df["date"].idxmax()
        return params_dict[idx]

    def query_vizier_param(self, param=None, radius=3, use_regex=False):
        """Looks for value of param in each Vizier table using exact match, regex, or wildcard."""
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius, verbose=False)
        else:
            tabs = self.vizier_tables

        if param is None:
            # Print all available columns
            cols = [tab.to_pandas().columns.tolist() for tab in tabs]
            logger.info(f"Choose parameter:\n{list(np.unique(sum(cols, [])))}")
        else:
            vals = {}
            matched_cols = []

            # Check for regex or exact match
            for i, tab in enumerate(tabs):
                columns = tab.columns
                if use_regex:
                    matches = [col for col in columns if re.search(param.replace('*', '.*'), col, re.IGNORECASE)]
                else:
                    matches = [param] if param in columns else []

                if matches:
                    k = tabs.keys()[i]
                    matched_cols.extend(matches)
                    for match in matches:
                        v = tab[match][0]  # Nearest match
                        v = np.nan if isinstance(v, np.ma.core.MaskedConstant) else v
                        vals[f"{k}:{match}"] = v

            if self.verbose:
                logger.info(f"Found {len(matched_cols)} references in Vizier using `{param}`.")
            return vals
            
            
    def query_simbad(self, target_name=None):
        """
        Query Simbad to get the object type of the target star.

        Returns
        -------
        res : SimbadResult
            The result of the query, if the target is resolved.
            Otherwise, None.
        """
        # See also: https://simbad.cds.unistra.fr/guide/otypes.htx
        Simbad.add_votable_fields("otype")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Try resolving the target star by name
            if self.target_name is None:
                self.target_name = input("What is the target's name?")
                self.tfop_info = self.query_tfop_info()
                self.parse_tfop_info()
                # ambiguous if query using coords
                # r = Simbad.query_region(self.target_coord, radius=1*u.arcmin)
                # if r is not None:
                #     return r
                # else:
                #     msg = f"Simbad cannot resolve {self.target_name}"
                #     msg += f" using coords:\n{self.target_coord.to_string('deg')}"
            for name in self.star_names:
                r = Simbad.query_object(name)
                if r is None:
                    continue
                else:
                    return r
            if self.verbose:
                msg = f"Simbad cannot resolve {self.target_name}"
                msg += f" using any of its names:\n{self.star_names}"
                logger.warning(msg)

    def get_simbad_obj_type(self):
        """
        Retrieves the object type of the target star from Simbad.

        Returns
        -------
        str or None
            The description of the object type if found, otherwise None.
        """
        # Query Simbad for the target star
        r = self.query_simbad()

        if r:
            # Extract the object type category
            category = r.to_pandas().squeeze()["OTYPE"]

            if len(category) >= 4:
                return category

            # Load Simbad object type descriptions
            df = pd.read_csv(simbad_obj_list_file)
            dd = df.query("Id==@category")

            if len(dd) > 0:
                # Retrieve the description and id
                desc = dd["Description"].squeeze()
                oid = dd["Id"].squeeze()

                # Check if the description contains 'binary' and print appropriate message
                if dd["Description"].str.contains("(?i)binary").any():
                    logger.info("***" * 15)
                    logger.info(
                        f"Simbad classifies {self.target_name} as {oid}={desc}!"
                    )
                    logger.info("***" * 15)
                else:
                    logger.info(
                        f"Simbad classifies {self.target_name} as {oid}={desc}!"
                    )

                return desc
        # Return None if no object type is found
        return None
    
    def query_binary_star_catalogs(self):
        self._query_star_catalog(VIZIER_KEYS_BINARY_STAR_CATALOG)                        
        
    def query_variable_star_catalogs(self):
        self._query_star_catalog(VIZIER_KEYS_VARIABLE_STAR_CATALOG)
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        all_tabs = self.query_vizier(verbose=False)
        # check for `var` in catalog title
        idx = [
            n if "var" in t._meta["description"] else False
            for n, t in enumerate(all_tabs)
        ]
        for i in idx:
            if i:
                tab = all_tabs[i]
                try:
                    s = tab.to_pandas().squeeze().str.decode("ascii").dropna()
                except Exception as e:
                    s = tab.to_pandas().squeeze().dropna()
                if len(s)>0:
                    print(f"\nSee also: {base_url}{tab._meta['name']}\n{s}")
                    self.variable_star = True
        
    def _query_star_catalog(self, catalog_keys):
        """
        Check for variable star flag in vizier and var in catalog title
        """
        base_url = "https://vizier.u-strasbg.fr/viz-bin/VizieR?-source="
        all_tabs = self.query_vizier(verbose=False)
        for key, tab in zip(all_tabs.keys(),all_tabs.values()):
            for ref, vkey in catalog_keys.items():
                if key==vkey:
                    d = tab.to_pandas().squeeze()
                    print(f"{ref} ({base_url}{key}):\n{d}")        
    
    def query_gaia_dr2_catalog(
        self, radius=None, version=2, return_nearest_xmatch=False, verbose=None
    ):
        """
        cross-match to Gaia DR2 catalog by angular separation
        position (accounting for proper motion) and brightess
        (comparing Tmag to Gmag whenever possible)

        Take caution:
        * phot_proc_mode=0 (i.e. “Gold” sources, see Riello et al. 2018)
        * astrometric_excess_noise_sig < 5
        * astrometric_gof_al < 20
        * astrometric_chi2_al
        * astrometric_n_good_obs_al
        * astrometric_primary_flag
        * duplicated source=0
        * visibility_periods_used
        * phot_variable_flag
        * flame_flags
        * priam_flags
        * phot_(flux)_excess_factor
        (a measure of the inconsistency between GBP, G, and GRP bands
        typically arising from binarity, crowdening and incomplete background
        modelling).

        Parameter
        ---------
        radius : float
            query radius in arcsec
        return_nearest_xmatch : bool
            return nearest single star if True else possibly more matches

        Returns
        -------
        tab : pandas.DataFrame
            table of star match(es)

        Notes:
        1. See column meaning here: https://mast.stsci.edu/api/v0/_c_a_o_mfields.html

        2. Gaia DR2 parallax has -0.08 mas offset (Stassun & Toress 2018,
        https://arxiv.org/pdf/1805.03526.pdf)

        3. quadratically add 0.1 mas to the uncertainty to account for systematics
        in the Gaia DR2 data (Luri+2018)

        4. Gmag has an uncertainty of 0.01 mag (Casagrande & VandenBerg 2018)

        From Carillo+2019:
        The sample with the low parallax errors i.e. 0 < f < 0.1,
        has distances derived from simply inverting the parallax

        Whereas, the sample with higher parallax errors i.e. f > 0.1
        has distances derived from a Bayesian analysis following Bailer-Jones (2015),
        where they use a weak distance prior (i.e. exponentially decreasing space
        density prior) that changes with Galactic latitude and longitude

        5. See also Gaia DR2 Cross-match for the celestial reference system (ICRS)
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_proc/ssec_cu3ast_proc_xmatch.html
        and
        https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_cali/ssec_cu3ast_cali_frame.html

        6. See https://github.com/tzdwi/TESS-Gaia and https://github.com/JohannesBuchner/nway
        and Salvato+2018 Appendix A for catalog matching problem: https://arxiv.org/pdf/1705.10711.pdf

        See also CDIPS gaia query:
        https://github.com/lgbouma/cdips/blob/master/cdips/utils/gaiaqueries.py

        See also bulk query:
        https://gea.esac.esa.int/archive-help/tutorials/python_cluster/index.html
        """
        if self.gaia_sources is not None:
            return self.gaia_sources.copy()
        
        radius = self.search_radius if radius is None else radius * u.arcsec
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            # silenced when verbose=False instead of None
            logger.info(
                f"""Querying Gaia DR2 catalog for ra,dec=({self.target_coord.to_string()}) within {radius:.2f}."""
            )
        # load gaia params for all TOIs
        tab = Catalogs.query_region(
            self.target_coord, radius=radius, catalog="Gaia", version=version
        ).to_pandas()
        # rename distance to separation because it is confusing
        tab = tab.rename(columns={"distance": "separation"})
        # convert from arcmin to arcsec
        tab["separation"] = tab["separation"].apply(
            lambda x: x * u.arcmin.to(u.arcsec)
        )
        errmsg = f"No gaia star within {radius}. Use radius>{radius}"
        assert len(tab) > 0, errmsg
        tab["source_id"] = tab.source_id.astype(int)
        # check if results from DR2 (epoch 2015.5)
        assert np.all(
            tab["ref_epoch"].isin([2015.5])
        ), "Epoch not 2015 (version<2?)"
        self.gaia_sources = tab
        
        if return_nearest_xmatch:
            nearest_match = tab.iloc[0]
            tplx = float(nearest_match["parallax"])
            if np.isnan(tplx) | (tplx < 0):
                # https://pyia.readthedocs.io/en/latest/#dealing-with-negative-parallaxes
                logger.info(f"Target parallax ({tplx} mas) is omitted!")
                tab["parallax"] = np.nan
        else:
            nstars = len(tab)
            idx1 = tab["parallax"] < 0
            tab.loc[idx1, "parallax"] = np.nan  # replace negative with nan
            idx2 = tab["parallax"].isnull()
            errmsg = f"No stars within radius={radius} have positive Gaia parallax!\n"
            if idx1.sum() > 0:
                errmsg += (
                    f"{idx1.sum()}/{nstars} stars have negative parallax!\n"
                )
            if idx2.sum() > 0:
                errmsg += f"{idx2.sum()}/{nstars} stars have no parallax!"
            assert len(tab) > 0, errmsg
        """
        FIXME: check parallax error here and apply corresponding distance calculation: see Note 1
        """
        if self.gaiaid is not None:
            errmsg = "Catalog does not contain target gaia id."
            assert np.any(tab["source_id"].isin([self.gaiaid])), errmsg

        # add gaia distance to target_coord
        # FIXME: https://docs.astropy.org/en/stable/coordinates/transforming.html
        gcoords = SkyCoord(
            ra=tab["ra"],
            dec=tab["dec"],
            unit="deg",
            frame="icrs",
            obstime="J2015.5",
        )
        # precess coordinate from Gaia DR2 epoch to J2000
        gcoords = gcoords.transform_to("icrs")
        if self.gaiaid is None:
            # find by nearest distance (for toiid or ticid input)
            idx = self.target_coord.separation(gcoords).argmin()
        else:
            # find by id match for gaiaDR2id input
            idx = tab.source_id.isin([self.gaiaid]).argmax()
        star = tab.loc[idx]
        # get distance from parallax
        if star["parallax"] > 0:
            target_dist = Distance(parallax=star["parallax"] * u.mas)
        else:
            target_dist = np.nan

        # redefine skycoord with coord and distance
        target_coord = SkyCoord(
            ra=self.target_coord.ra,
            dec=self.target_coord.dec,
            distance=target_dist,
        )
        self.target_coord = target_coord

        nsources = len(tab)
        if return_nearest_xmatch or (nsources == 1):
            if nsources > 1:
                msg=f"There are {nsources} gaia sources within {radius}."
                logger.info(msg)
            target = tab.iloc[0]
            if self.gaiaid is not None:
                id = int(target["source_id"])
                msg = f"Nearest match ({id}) != {self.gaiaid}"
                assert int(self.gaiaid) == id, msg
            else:
                self.gaiaid = int(target["source_id"])
            self.gaia_params = target
            self.gmag = target["phot_g_mean_mag"]
            ens = target["astrometric_excess_noise_sig"]
            if ens >= 5:
                msg = f"astrometric_excess_noise_sig={ens:.2f} (>5 hints binarity).\n"
                logger.info(msg)
            gof = target["astrometric_gof_al"]
            if gof >= 20:
                msg = f"astrometric_gof_al={gof:.2f} (>20 hints binarity)."
                logger.info(msg)
            if (ens >= 5) or (gof >= 20):
                logger.info("See https://arxiv.org/pdf/1804.11082.pdf\n")
            delta = np.hypot(target["pmra"], target["pmdec"])
            if abs(delta) > 10:
                msg = "High proper-motion star:\n"
                msg+=f"(pmra,pmdec)=({target['pmra']:.2f},{target['pmdec']:.2f}) mas/yr"
                logger.info(msg)
            if target["visibility_periods_used"] < 6:
                msg = "visibility_periods_used<6 so no astrometric solution\n"
                msg += "See https://arxiv.org/pdf/1804.09378.pdf\n"
                logger.info(msg)
            ruwe = list(self.query_vizier_param("ruwe").values())
            if len(ruwe) > 0 and ruwe[0] > 1.4:
                msg = f"RUWE={ruwe[0]:.1f}>1.4 means target is non-single or otherwise problematic for the astrometric solution.\n"
                msg += "See also https://arxiv.org/pdf/2404.14127"
                logger.info(msg)
            return target  # return series of len 1
        else:
            # if self.verbose:
            #     d = self.get_nearby_gaia_sources()
            #     print(d)
            return tab  # return dataframe of len 2 or more
    
    def __repr__(self):
        """Override to print a readable string representation of class"""
        # params = signature(self.__init__).parameters
        # val = repr(getattr(self, key))

        included_args = [
            # ===target attributes===
            "target_name",
            # "toiid",
            # "ctoiid",
            # "ticid",
            # "epicid",
            "gaiaDR2id",
            "ra_deg",
            "dec_deg",
            "target_coord",
            "search_radius",
        ]
        args = []
        for key in self.__dict__.keys():
            val = self.__dict__.get(key)
            if key in included_args:
                if key == "target_coord":
                    # format coord
                    coord = self.target_coord.to_string("decimal")
                    args.append(f"{key}=({coord.replace(' ',',')})")
                elif val is not None:
                    args.append(f"{key}={val}")
        args = ", ".join(args)
        return f"{type(self).__name__}({args})"
    
def get_tois_data(
    clobber=False,
    outdir=DATA_PATH,
    verbose=False,
    remove_FP=True,
    remove_known_planets=False
):
    """Download TOI list from TESS Alert/TOI Release.

    Parameters
    ----------
    clobber : bool
        re-download table and save as csv file
    outdir : str
        download directory location
    verbose : bool
        print texts

    Returns
    -------
    d : pandas.DataFrame
        TOI table as dataframe
    """
    dl_link = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    fp = Path(outdir, "TOIs.csv")
    if not Path(outdir).exists():
        Path(outdir).makedir()

    if not fp.exists() or clobber:
        df = pd.read_csv(dl_link)  # , dtype={'RA': float, 'Dec': float})
        msg = f"Downloading {dl_link}\n"

        #add coordinates in deg
        coords = SkyCoord(df[['RA','Dec']].values, unit=('hourangle','degree'))
        df['ra_deg'] = coords.ra.deg
        df['dec_deg'] = coords.dec.deg
        df.to_csv(fp, index=False)
        print("Saved: ", fp)
    else:
        df = pd.read_csv(fp).drop_duplicates()
        msg = f"Loaded: {fp}\n"
    assert len(df) > 1000, f"{fp} likely has been overwritten!"

    # remove False Positives
    if remove_FP:
        df = df[df["TFOPWG Disposition"] != "FP"]
        msg += "TOIs with TFPWG disposition==FP are removed.\n"
    if remove_known_planets:
        planet_keys = [
            "HD",
            "GJ",
            "LHS",
            "XO",
            "Pi Men", 
            "WASP",
            "SWASP",
            "HAT",
            "HATS",
            "KELT",
            "TrES",
            "QATAR",
            "CoRoT",
            "K2",  # , "EPIC"
            "Kepler",  # "KOI"
        ]
        keys = []
        for key in planet_keys:
            idx = ~np.array(
                df["Comments"].str.contains(key).tolist(), dtype=bool
            )
            df = df[idx]
            if idx.sum() > 0:
                keys.append(key)
        msg += f"{keys} planets are removed.\n"
    msg += f"Saved: {fp}\n"
    if verbose:
        print(msg)
    return df.sort_values("TOI").reset_index(drop=True)

def get_nexsci_data(table_name="ps", method="Transit", outdir=DATA_PATH, clobber=False):
    """
    ps: self-consistent set of parameters
    pscomppars: a more complete, though not necessarily self-consistent set of parameters
    See also 
    https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
    """
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
    except Exception as e:
        print(e)
    assert table_name in ["ps", "pscomppars"]
    url = "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html"
    print("Column definitions: ", url)
    fp = Path(outdir, f'nexsci_{table_name}.csv')        
    if not fp.exists() or clobber:
        #pstable combines data from the Confirmed Planets and Extended Planet Parameters tables
        print(f"Downloading NExSci {table_name} table...")
        tab = NasaExoplanetArchive.query_criteria(table=table_name, 
                                                  where=f"discoverymethod like '{method}'")
        df = tab.to_pandas()
        df.to_csv(fp, index=False)
        print("Saved: ", fp)
    else:
        df = pd.read_csv(fp)
        print("Loaded: ", fp)
    return df

def get_mamajek_table(clobber=False, verbose=True, data_loc=DATA_PATH):
    """
    """
    fp = join(data_loc, "mamajek_table.csv")
    if not exists(fp) or clobber:
        url = "http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt"
        # cols="SpT Teff logT BCv Mv logL B-V Bt-Vt G-V U-B V-Rc V-Ic V-Ks J-H H-Ks Ks-W1 W1-W2 W1-W3 W1-W4 Msun logAge b-y M_J M_Ks Mbol i-z z-Y R_Rsun".split(' ')
        df = pd.read_csv(
            url,
            skiprows=21,
            skipfooter=524,
            delim_whitespace=True,
            engine="python",
        )
        # tab = ascii.read(url, guess=None, data_start=0, data_end=124)
        # df = tab.to_pandas()
        # replace ... with NaN
        df = df.replace(["...", "....", "....."], np.nan)
        # replace header
        # df.columns = cols
        # drop last duplicate column
        df = df.drop(df.columns[-1], axis=1)
        # df['#SpT_num'] = range(df.shape[0])
        # df['#SpT'] = df['#SpT'].astype('category')

        # remove the : type in M_J column
        df["M_J"] = df["M_J"].apply(lambda x: str(x).split(":")[0])
        # convert columns to float
        for col in df.columns:
            if col == "#SpT":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(float)
            # if col=='SpT':
            #     df[col] = df[col].astype('categorical')
            # else:
            #     df[col] = df[col].astype(float)
        df.to_csv(fp, index=False)
        print(f"Saved: {fp}")
    else:
        df = pd.read_csv(fp)
        if verbose:
            print(f"Loaded: {fp}")
    return df


def interpolate_mamajek_table(
        df,
        input_col="BP-RP",
        output_col="Teff",
        nsamples=int(1e4),
        return_samples=False,
        plot=False,
        clobber=False,
        verbose=True
    ):
        """
        Interpolate spectral type from Mamajek table from
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        based on observables Teff and color indices.
        c.f. self.query_vizier_param("SpT")

        Parameters
        ----------
        columns : list
            column names of input parameters
        nsamples : int
            number of Monte Carlo samples (default=1e4)
        clobber : bool (default=False)
            re-download Mamajek table

        Returns
        -------
        spt: str
            interpolated spectral type

        Notes:
        It may be good to check which color index yields most accurate result

        Check sptype from self.query_simbad()
        """
        df = get_mamajek_table(clobber=clobber, verbose=verbose)
        
        # B-V color index
        bprp_color = df["BPRP"]
        ubprp_color = bprp_color.std()
        s_bprp_color = (
            bprp_color + np.random.randn(nsamples) * ubprp_color
        )  # Monte Carlo samples

        # Interpolate
        interp = NearestNDInterpolator(
            df[input_col].values, df[output_col].values, rescale=False
        )
        samples = interp(s_bprp_color)
        # encode category
        spt_cats = pd.Series(samples, dtype="category")  # .cat.codes
        spt = spt_cats.mode().values[0]
        if plot:
            nbins = np.unique(samples)
            pl.hist(samples, bins=nbins)
        if return_samples:
            return spt, samples
        else:
            return spt


def flatten_list(lol):
    """flatten list of list (lol)"""
    return list(itertools.chain.from_iterable(lol))


def get_tfop_info(target_name: str) -> dict:
    base_url = "https://exofop.ipac.caltech.edu/tess"
    url = f"{base_url}/target.php?id={target_name.replace(' ','')}&json"
    response = urlopen(url)
    assert response.code == 200, "Failed to get data from ExoFOP-TESS"
    try:
        data_json = json.loads(response.read())
        return data_json
    except Exception as e:
        print(e)
        raise ValueError(f"No TIC data found for {target_name}")


def get_params_from_tfop(tfop_info, name="planet_parameters", idx=None):
    params_dict = tfop_info.get(name)
    if idx is None:
        key = "pdate" if name == "planet_parameters" else "sdate"
        # get the latest parameter based on upload date
        dates = []
        for d in params_dict:
            t = d.get(key)
            dates.append(t)
        df = pd.DataFrame({"date": dates})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        idx = df["date"].idxmax()
    return params_dict[idx]


def get_tic_id(target_name: str) -> int:
    return int(get_tfop_info(target_name)["basic_info"]["tic_id"])


def get_nexsci_data(table_name="ps", clobber=False):
    """
    ps: self-consistent set of parameters
    pscomppars: a more complete, though not necessarily self-consistent set of parameters
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html"
    print("Column definitions: ", url)
    fp = Path("../data/",f"nexsci_{table_name}.csv")
    if not fp.exists() or clobber:
        print(f"Downloading NExSci {table_name} table...")
        nexsci_tab = NasaExoplanetArchive.query_criteria(table=table_name, where="discoverymethod like 'Transit'")
        df_nexsci = nexsci_tab.to_pandas()
        df_nexsci.to_csv(fp, index=False)
        print("Saved: ", fp)
    else:
        df_nexsci = pd.read_csv(fp)
        print("Loaded: ", fp)
    return df_nexsci

def plot_age_posterior(name, Prot, Teff, age_posterior, age_stat, age_grid, xlim=[0,1000]):
    fig, ax = pl.subplots()
    
    # add vertical lines
    assert np.isfinite(list(age_stat.values())).all()
    x1 = int(age_stat['-1sigma'])
    med = int(age_stat['median'])
    x2 = int(age_stat['+1sigma'])    
    lbl = f"Age = {med} + {x2} - {x1} Myr"
    
    ax.plot(age_grid, 1e3*age_posterior, c='k', lw=1, marker='.',label=lbl)
    ax.update({
        'xlabel': 'Age [Myr]',
        'ylabel': 'Probability ($10^{-3}\,$Myr$^{-1}$)',
        'title': f'{name} (Prot = {Prot:.2f}d, Teff = {int(Teff)}K)',
        'xlim': xlim
    })
    
    for i,v in zip(['-1sigma','median','+1sigma'],[med-x1,med,med+x2]):
        ls = '-' if i=='median' else '--'
        ax.axvline(v, ls=ls)
    ax.legend()
    return fig