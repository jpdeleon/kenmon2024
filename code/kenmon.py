import json
import itertools
from urllib.request import urlopen
import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
from pathlib import Path
from pprint import pprint
import astropy.units as u
from astropy.coordinates import SkyCoord, Distance, Galactocentric
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy.table import Table
from astroquery.vizier import Vizier

current_dir = Path(__file__).parent
DATA_PATH = current_dir.parent / "data"
    
def get_tois(
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
        #add previously querried Gaia DR3 ids
        tois = pd.read_csv(f'{outdir}/TOIs_with_Gaiaid.csv')
        df = pd.merge(df, tois, on='TOI', how='outer')
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
    

class Target:
    def __init__(self, ra_deg, dec_deg, gaiaDR2id=None, verbose=True):
        self.gaiaid = gaiaDR2id  # e.g. Gaia DR2 5251470948229949568
        self.ra = ra_deg
        self.dec = dec_deg
        self.target_coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
        self.verbose = verbose
        self.vizier_tables = None
        self.search_radius = 1 * u.arcmin

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
            tables = self.vizier_tables.filled(fill_value)
        return tables

    def query_vizier_param(self, param=None, radius=3):
        """looks for value of param in each vizier table"""
        if self.vizier_tables is None:
            tabs = self.query_vizier(radius=radius, verbose=False)
        else:
            tabs = self.vizier_tables

        if param is not None:
            idx = [param in tab.columns for tab in tabs]
            vals = {}
            for i in np.argwhere(idx).flatten():
                k = tabs.keys()[int(i)]
                v = tabs[int(i)][param][0] #nearest match
                if isinstance(v, np.ma.core.MaskedConstant):
                    v = np.nan
                vals[k] = v
            if self.verbose:
                print(f"Found {sum(idx)} references in Vizier with `{param}`.")
            return vals
        else:
            #print all available keys
            cols = [tab.to_pandas().columns.tolist() for tab in tabs]
            print(f"Choose parameter:\n{list(np.unique(flatten_list(cols)))}")
    
    def __repr__(self):
        """Override to print a readable string representation of class"""
        # params = signature(self.__init__).parameters
        # val = repr(getattr(self, key))

        included_args = [
            # ===target attributes===
            # "name",
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