import {
    useState,
} from "react";

import {
    DatePicker,
    Select,
} from "antd";
import dayjs from "dayjs";

import useSearchStore from "../../SearchContext";
import styles from "./index.module.css";
import {
    isValidDateRange,
    TIME_RANGE_OPTION,
    TIME_RANGE_OPTION_NAMES,
    timeRangeOptionDayJsMap,
} from "./utils";


const DEFAULT_TIME_RANGE = TIME_RANGE_OPTION.TODAY;

/**
 * Renders controls for selecting a time range for queries. By default, the component is
 * a select dropdown with a list of preset time ranges. If the user selects "Custom",
 * a date range picker is also displayed.
 *
 * @return
 */
const TimeRangeInput = () => {
    const updateTimeRange = useSearchStore((state) => state.updateTimeRange);
    const [selectedOption, setSelectedOption] = useState<TIME_RANGE_OPTION>(DEFAULT_TIME_RANGE);

    const handleSelectChange = (timeRangeOption: TIME_RANGE_OPTION) => {
        setSelectedOption(timeRangeOption);
        if (timeRangeOption !== TIME_RANGE_OPTION.CUSTOM) {
            const dayJsRange = timeRangeOptionDayJsMap[timeRangeOption];
            updateTimeRange(dayJsRange);
        }
    };

    const handleRangePickerChange = (
        dates: [dayjs.Dayjs | null, dayjs.Dayjs | null] | null
    ) => {
        if (!isValidDateRange(dates)) {
            return;
        }
        updateTimeRange(dates);
    };

    return (
        <div
            className={styles["time-range-input-container"]}
        >
            <Select
                defaultValue={DEFAULT_TIME_RANGE}
                listHeight={300}
                options={TIME_RANGE_OPTION_NAMES.map((option) => ({label: option, value: option}))}
                popupMatchSelectWidth={false}
                size={"large"}
                variant={"filled"}
                onChange={handleSelectChange}/>
            {selectedOption === TIME_RANGE_OPTION.CUSTOM && (
                <DatePicker.RangePicker
                    className={styles["range-picker"] || ""}
                    showNow={true}
                    showTime={true}
                    size={"large"}
                    onChange={(dates) => {
                        handleRangePickerChange(dates);
                    }}/>
            )}
        </div>
    );
};

export default TimeRangeInput;
