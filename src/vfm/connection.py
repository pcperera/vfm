from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import Token
from msal import PublicClientApplication
import pandas as pd

TENANT_ID = "d9f7afd7-be4d-4a0f-9d22-10ebc305b615"
CLIENT_ID = "d0b33cfc-f288-466e-84d0-a9508387b304"
CDF_CLUSTER = "api"
COGNITE_PROJECT = "production-optimisation-rd"
BASE_URL = f"https://{CDF_CLUSTER}.cognitedata.com"
SCOPES = [f"https://{CDF_CLUSTER}.cognitedata.com/.default"]
AUTHORITY_HOST_URI = "https://login.microsoftonline.com"
AUTHORITY_URI = AUTHORITY_HOST_URI + "/" + TENANT_ID
PORT = 53000

class Connection:

    def __init__(self):
        self.__public_client_app = PublicClientApplication(client_id=CLIENT_ID, authority=AUTHORITY_URI)

    def get_client(self):
        creds = self.__public_client_app.acquire_token_interactive(scopes=SCOPES, port=PORT)
        cnf = ClientConfig(client_name="my-special-client", project=COGNITE_PROJECT,
                           credentials=Token(creds["access_token"]), base_url=BASE_URL)
        client = CogniteClient(cnf)
        return client
    

    def get_data(
        self,
        client: CogniteClient,
        wells: list[str],
        start: int = None,
        end: int = None,
    ) -> pd.DataFrame:
        """
        Retrieve well test data for multiple wells and return a single DataFrame.
        """

        all_well_dfs = []

        for well in wells:
            mapping = {}

            # # MPFM
            # mpfm_wc = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WC_MPFM", limit=None, start=start, end=end)
            # mpfm_oil_rate = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_OIL_RATE_MPFM", limit=None, start=start, end=end)
            # mpfm_gas_rate = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_GAS_RATE_MPFM", limit=None, start=start, end=end)


            # # Well test
            # wt_whp = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WELLHEAD_PRESSURE", limit=None, start=start, end=end)
            # wt_wht = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WELLHEAD_TEMPERATURE", limit=None, start=start, end=end)
            # wt_dhp = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_DOWNHOLE_PRESSURE", limit=None, start=start, end=end)
            # wt_dht = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_DOWNHOLE_TEMPERATURE", limit=None, start=start, end=end)
            # wt_gas_rate_test_sperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_GAS_RATE_TEST_SEPARATOR", limit=None, start=start, end=end)
            # wt_water_rate_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_QW_TEST_SEPARATOR", limit=None, start=start, end=end)
            # wt_oil_rate_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_OIL_RATE_TEST_SEPARATOR", limit=None, start=start, end=end)
            # wt_gor_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_GOR_TEST_SEPARATOR", limit=None, start=start, end=end)
            # wt_wc_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WC_TEST_SEPARATOR", limit=None, start=start, end=end)
            # wt_choke_downstream_pressure = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_DOWNSTREAM_CHOKE_PRES", limit=None, start=start, end=end)
            # wt_choke_position = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_CHOKE_POS", limit=None, start=start, end=end)


            mapping["WHP"]   = f"{well}_WELL_TEST_WELLHEAD_PRESSURE"
            mapping["WHT"]   = f"{well}_WELL_TEST_WELLHEAD_TEMPERATURE"
            mapping["DHP"]   = f"{well}_WELL_TEST_DOWNHOLE_PRESSURE"
            mapping["DHT"]   = f"{well}_WELL_TEST_DOWNHOLE_TEMPERATURE"
            mapping["CHOKE"] = f"{well}_WELL_TEST_CHOKE_POS"
            mapping["DCP"]   = f"{well}_WELL_TEST_DOWNSTREAM_CHOKE_PRES"

            # Well test rates (naming convention preserved)
            mapping["QO_WELL_TEST"] = f"LER_Q_OIL_WELL_TEST_WELL{well[1:]}"
            mapping["QG_WELL_TEST"] = f"LER_Q_GAS_WELL_TEST_WELL{well[1:]}"
            mapping["QW_WELL_TEST"] = f"LER_Q_WATER_WELL_TEST_WELL{well[1:]}"

            # MPFM
            mapping["QO_MPFM"] = f"{well}_WELL_TEST_OIL_RATE_MPFM"
            mapping["QG_MPFM"] = f"{well}_WELL_TEST_GAS_RATE_MPFM"
            mapping["WC_MPFM"] = f"{well}_WELL_TEST_WC_MPFM"

            external_ids = list(mapping.values())

            # Retrieve all time series for this well in one call
            df = client.time_series.data.retrieve_dataframe(
                external_id=external_ids,
                start=start,
                end=end,
            )

            if df.empty:
                print(f"No data returned for well {well}")
                continue

            # Rename columns to consistent lowercase feature names
            df = df.rename(columns={mapping[k]: k.lower() for k in mapping})

            # Add well identifier
            df["well_id"] = well

            # Ensure time ordering
            df.sort_index(inplace=True)

            all_well_dfs.append(df)

        if not all_well_dfs:
            return pd.DataFrame()

        # Concatenate all wells
        data = pd.concat(all_well_dfs, axis=0)

        return data

    # def get_data(self, client: CogniteClient, well: str = 'W06', start:int = None, end: int = None):
    #     # # MPFM
    #     # mpfm_wc = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WC_MPFM", limit=None, start=start, end=end)
    #     # mpfm_oil_rate = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_OIL_RATE_MPFM", limit=None, start=start, end=end)
    #     # mpfm_gas_rate = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_GAS_RATE_MPFM", limit=None, start=start, end=end)


    #     # # Well test
    #     # wt_whp = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WELLHEAD_PRESSURE", limit=None, start=start, end=end)
    #     # wt_wht = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WELLHEAD_TEMPERATURE", limit=None, start=start, end=end)
    #     # wt_dhp = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_DOWNHOLE_PRESSURE", limit=None, start=start, end=end)
    #     # wt_dht = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_DOWNHOLE_TEMPERATURE", limit=None, start=start, end=end)
    #     # wt_gas_rate_test_sperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_GAS_RATE_TEST_SEPARATOR", limit=None, start=start, end=end)
    #     # wt_water_rate_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_QW_TEST_SEPARATOR", limit=None, start=start, end=end)
    #     # wt_oil_rate_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_OIL_RATE_TEST_SEPARATOR", limit=None, start=start, end=end)
    #     # wt_gor_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_GOR_TEST_SEPARATOR", limit=None, start=start, end=end)
    #     # wt_wc_test_seperator = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_WC_TEST_SEPARATOR", limit=None, start=start, end=end)
    #     # wt_choke_downstream_pressure = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_DOWNSTREAM_CHOKE_PRES", limit=None, start=start, end=end)
    #     # wt_choke_position = client.time_series.data.retrieve_dataframe(external_id=f"{well}_WELL_TEST_CHOKE_POS", limit=None, start=start, end=end)

    #     mapping = {}
    #     mapping["WHP"] = f"{well}_WELL_TEST_WELLHEAD_PRESSURE"
    #     mapping["WHT"] = f"{well}_WELL_TEST_WELLHEAD_TEMPERATURE"
    #     mapping["DHP"] = f"{well}_WELL_TEST_DOWNHOLE_PRESSURE"
    #     mapping["DHT"] = f"{well}_WELL_TEST_DOWNHOLE_TEMPERATURE"
    #     mapping["CHOKE"] = f"{well}_WELL_TEST_CHOKE_POS"
    #     mapping["DCP"] = f"{well}_WELL_TEST_DOWNSTREAM_CHOKE_PRES"

    #     mapping["QO_WELL_TEST"] = f"LER_Q_OIL_WELL_TEST_WELL{well[1:]}"
    #     mapping["QG_WELL_TEST"] = f"LER_Q_GAS_WELL_TEST_WELL{well[1:]}"
    #     mapping["QW_WELL_TEST"] = f"LER_Q_WATER_WELL_TEST_WELL{well[1:]}"

    #     mapping["QO_MPFM"] = f"{well}_WELL_TEST_OIL_RATE_MPFM"
    #     mapping["QG_MPFM"] = f"{well}_WELL_TEST_GAS_RATE_MPFM"
    #     mapping["WC_MPFM"] = f"{well}_WELL_TEST_WC_MPFM"

    #     # xids = [mapping[item] for item in
    #     #         ["WHP", "DHP", "CHOKE", "DCP", "QO_MPFM", "GOR_MPFM", "WC_MPFM", "WHT", "WELL_TEST_OIL",
    #     #          "WELL_TEST_WATER", "WELL_TEST_GAS", "DHT"]]

    #     external_ids = list(mapping.values())
    #     data = client.time_series.data.retrieve_dataframe(external_id=external_ids, start=start, end=end)
    #     data = data.rename(columns={mapping[item]: item.lower() for item in mapping})
        
    #     # data = data.rename(columns={"whp": Feature.WELL_HEAD_PRESSURE.value, "dhp": Feature.DOWN_HOLE_PRESSURE.value, "dcp": Feature.DOWNSTREAM_CHOKE_PRESSURE.value,
    #     #                             "wht": Feature.WELL_HEAD_TEMPERATURE.value, "dht": Feature.DOWN_HOLE_TEMPERATURE.value,
    #     #                             "qo_mpfm": Feature.QO_MPFM.value})

    #     data["well_id"] = well
    #     data.sort_index(inplace=True)  # Ensure chronological order
    #     return data
