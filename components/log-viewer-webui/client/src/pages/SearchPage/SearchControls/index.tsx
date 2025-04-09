import {Input} from "antd";

import useSearchStore from "../SearchContext";
import styles from "./index.module.css";
import SearchButton from "./SearchButton";
import TimeRangeInput from "./TimeRangeInput";

/**
 * Renders controls for submitting queries.
 *
 * @return
 */
const SearchControls = () => {
    const queryString = useSearchStore((state) => state.queryString);
    const updateQueryString = useSearchStore((state) => state.updateQueryString);

    return (
        <div className={styles["search-controls-container"]}>
            <Input
                placeholder={"Enter your query"}
                size={"large"}
                value={queryString}
                onChange={(e) => {
                    updateQueryString(e.target.value);
                }}/>
            <TimeRangeInput/>
            <SearchButton/>
        </div>
    );
};

export default SearchControls;
