import {useEffect} from "react";

import {useQuery} from "@tanstack/react-query";
import {message} from "antd";

import useSearchStore from "../../SearchState/index";
import {fetchDatasetNames} from "./sql";


/**
 * Custom hook for retrieving datasets from database and managing dataset selection.
 *
 * @return
 */
const useDatasetSelect = () => {
    const dataset = useSearchStore((state) => state.selectDataset);
    const updateDataset = useSearchStore((state) => state.updateSelectDataset);
    const [messageApi, contextHolder] = message.useMessage();
    const {data: datasets, isPending, isSuccess, error} = useQuery({
        queryKey: ["datasets"],
        queryFn: fetchDatasetNames,
    });

    // Update the selected dataset to the first dataset in the response. The dataset is only
    // updated if it isn't already set (i.e., on initial response).
    useEffect(() => {
        if (isSuccess) {
            if ("undefined" !== typeof datasets[0] && null === dataset) {
                updateDataset(datasets[0]);
            }
        }
    }, [isSuccess, datasets, dataset, updateDataset]);

    // Display error message if the query fails since user querying is disabled if no datasets.
    useEffect(() => {
        if (error) {
            messageApi.error({
                key: "fetchError",
                content: "Error fetching datasets.",
            });
        }
    }, [error, messageApi]);

    // Display warning message if response empty since user querying is disabled if no datasets.
    useEffect(() => {
        if (isSuccess && datasets && 0 === datasets.length) {
            messageApi.warning({
                key: "noData",
                content: "No data has been ingested. Please ingest data to search.",
            });
            updateDataset(null);
        }
    }, [datasets, isSuccess, messageApi, updateDataset]);

    const handleDatasetChange = (value: string) => {
        updateDataset(value);
    };

    return {
        contextHolder,
        datasets,
        dataset,
        handleDatasetChange,
        isPending,
    };
};

export default useDatasetSelect;
