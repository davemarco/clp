import {useEffect} from "react";
import {useQuery} from "@tanstack/react-query";
import {message} from "antd";

/**
 * Generic custom hook for retrieving options and managing selection.
 *
 * @param {Function} fetchFn - Function to fetch options.
 * @param {string} queryKey - Query key for react-query.
 * @return {object} Hook state and handlers for selection.
 */
const useSelect = (fetchFn: () => Promise<string[]>, queryKey: string) => {
    const [selected, setSelected] = useState<string | null>(null);
    const [messageApi, contextHolder] = message.useMessage();
    const {data: options, isPending, isSuccess, error} = useQuery({
        queryKey: [queryKey],
        queryFn: fetchFn,
    });

    useEffect(() => {
        if (isSuccess) {
            if (options && options[0] !== undefined && selected === null) {
                setSelected(options[0]);
            }
        }
    }, [isSuccess, options, selected, setSelected]);

    useEffect(() => {
        if (error) {
            messageApi.error({key: `fetch${queryKey}Error`, content: `Error fetching ${queryKey}.`});
        }
    }, [error, messageApi, queryKey]);

    useEffect(() => {
        if (isSuccess && options && options.length === 0) {
            messageApi.warning({
                key: `no${queryKey}`,
                content: `No ${queryKey} found. Please check your database.`,
            });
            setSelected(null);
        }
    }, [options, isSuccess, messageApi, queryKey]);

    const handleChange = (value: string) => {
        setSelected(value);
    };

    return {
        contextHolder,
        options,
        selected,
        handleChange,
        isPending,
    };
};

export default useSelect;
