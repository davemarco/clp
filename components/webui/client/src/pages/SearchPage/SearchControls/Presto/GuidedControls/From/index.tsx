import {useState, useEffect} from "react";
import {useQuery} from "@tanstack/react-query";
import {Select, message} from "antd";
import InputLabel from "../../../../../../components/InputLabel";
import guidedGrid from "./index.module.css";
import {fetchPrestoTableNames} from "./sql";


/**
 * Renders a table selector component that allows users to select from available Presto tables.
 *
 * @return
 */
const From = () => {
    const [selectedTable, setSelectedTable] = useState<string | null>(null);
    const [messageApi, contextHolder] = message.useMessage();

    const {data, isPending, isSuccess, error} = useQuery({
        queryKey: ["prestoTables"],
        queryFn: fetchPrestoTableNames,
    });

    // Update the selected dataset to the first dataset in the response. The dataset is only
    // updated if it isn't already set (i.e., on initial response).
    useEffect(() => {
        if (isSuccess) {
            if (typeof data?.[0] !== "undefined" && selectedTable === null) {
                setSelectedTable(data[0]);
            }
        }
    }, [
        isSuccess,
        data,
        selectedTable,
        setSelectedTable
    ]
    );

    // Display error message if the query fails since user querying is disabled if no tables.
    useEffect(() => {
        if (error) {
            messageApi.error({key: "fetchTablesError", content: "Error fetching tables from Presto."});
        }
    }, [error, messageApi]);

    // Display warning message if response empty since user querying is disabled if no tables.
    useEffect(() => {
        if (isSuccess && data && data.length === 0) {
            messageApi.warning({
                key: "noTables",
                content: "No tables found in Presto. Please check your database.",
            });
            setSelectedTable(null);
        }
    }, [data, isSuccess, messageApi]);

    const handleTableChange = (value: string) => {
        setSelectedTable(value);
    };

    return (
        <div className={guidedGrid["from"]}>
            {contextHolder}
            <InputLabel>FROM</InputLabel>
            <Select
                loading={isPending}
                options={(data || []).map((table) => ({label: table, value: table}))}
                placeholder={"Select table"}
                showSearch={true}
                size={"large"}
                value={selectedTable}
                onChange={handleTableChange}
                style={{width: "100%"}}
            />
        </div>
    );
};

export default From;
