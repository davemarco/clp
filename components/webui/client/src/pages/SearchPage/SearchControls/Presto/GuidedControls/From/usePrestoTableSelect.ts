import {useEffect} from "react";
import {useQuery} from "@tanstack/react-query";
import {message} from "antd";
import {fetchPrestoTableNames} from "./sql";

/**
 * Custom hook for retrieving Presto tables and managing table selection.
 *
 * @return {object} Hook state and handlers for Presto table selection.
 */
const usePrestoTableSelect = () => {
    const [selectedTable, setSelectedTable] = useState<string | null>(null);
    const [messageApi, contextHolder] = message.useMessage();
    const {data: tables, isPending, isSuccess, error} = useQuery({
        queryKey: ["prestoTables"],
        queryFn: fetchPrestoTableNames,
    });

    useEffect(() => {
        if (isSuccess) {
            if (tables && tables[0] !== undefined && selectedTable === null) {
                setSelectedTable(tables[0]);
            }
        }
    }, [isSuccess, tables, selectedTable, setSelectedTable]);

    useEffect(() => {
        if (error) {
            messageApi.error({key: "fetchTablesError", content: "Error fetching tables from Presto."});
        }
    }, [error, messageApi]);

    useEffect(() => {
        if (isSuccess && tables && tables.length === 0) {
            messageApi.warning({
                key: "noTables",
                content: "No tables found in Presto. Please check your database.",
            });
            setSelectedTable(null);
        }
    }, [tables, isSuccess, messageApi]);

    const handleTableChange = (value: string) => {
        setSelectedTable(value);
    };

    return {
        contextHolder,
        tables,
        selectedTable,
        handleTableChange,
        isPending,
    };
};

export default usePrestoTableSelect;
