import {Select} from "antd";
import InputLabel from "../../../../../../components/InputLabel";
import guidedGrid from "../index.module.css";
import useDatasetSelect from "../../../Dataset/useDatasetSelect";


/**
 * Renders a table selector component that allows users to select from available Presto tables.
 *
 * @return
 */
const From = () => {

    const {
        contextHolder,
        datasets,
        dataset,
        handleDatasetChange,
        isPending,
    } = useDatasetSelect();

    return (
        <div className={guidedGrid["from"]}>
            <InputLabel>FROM</InputLabel>
            <Select
                options={(datasets || []).map((table) => ({label: table, value: table}))}
                placeholder={"Select table"}
                size={"middle"}
                value={dataset}
                style={{height: "100%", width: "100%"}}
                onChange={handleDatasetChange}
            />
        </div>
    );
};

export default From;
