import {Select} from "antd";

import InputLabel from "../../../../components/InputLabel";
import useSearchStore from "../../SearchState/index";
import {SEARCH_UI_STATE} from "../../SearchState/typings";
import styles from "./index.module.css";
import useDatasetSelect from "./useDatasetSelect";


/**
 * Renders a dataset selector component that allows users to select from available datasets.
 *
 * @return
 */
const Dataset = () => {

    const {
        dataset,
        datasets,
        isPending,
        contextHolder,
        handleDatasetChange,
    } = useDatasetSelect();

    const searchUiState = useSearchStore((state) => state.searchUiState);

    return (
        <div className={styles["datasetContainer"]}>
            <InputLabel>Dataset</InputLabel>
            {contextHolder}
            <Select
                className={styles["select"] || ""}
                loading={isPending}
                options={(datasets || []).map((option) => ({label: option, value: option}))}
                placeholder={"None"}
                showSearch={true}
                size={"middle"}
                value={dataset}
                disabled={
                    searchUiState === SEARCH_UI_STATE.QUERY_ID_PENDING ||
                    searchUiState === SEARCH_UI_STATE.QUERYING
                }
                onChange={handleDatasetChange}/>
        </div>
    );
};

export default Dataset;
