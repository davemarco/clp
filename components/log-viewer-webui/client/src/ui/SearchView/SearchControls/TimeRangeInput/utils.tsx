import dayjs from "dayjs";


/**
 * Time range options.
 */
enum TIME_RANGE_OPTION {
    LAST_15_MINUTES = "Last 15 Minutes",
    LAST_HOUR = "Last Hour",
    TODAY = "Today",
    YESTERDAY = "Yesterday",
    LAST_7_DAYS = "Last 7 Days",
    LAST_30_DAYS = "Last 30 Days",
    MONTH_TO_DATE = "Month to Date",
    CUSTOM = "Custom",
}

/* eslint-disable no-magic-numbers */
const timeRangeOptionDayJsMap: Record<TIME_RANGE_OPTION, [dayjs.Dayjs, dayjs.Dayjs]> = {
    [TIME_RANGE_OPTION.LAST_15_MINUTES]: [dayjs().subtract(15, "minute"),
        dayjs()],
    [TIME_RANGE_OPTION.LAST_HOUR]: [dayjs().subtract(1, "hour"),
        dayjs()],
    [TIME_RANGE_OPTION.TODAY]: [dayjs().startOf("day"),
        dayjs().endOf("day")],
    [TIME_RANGE_OPTION.YESTERDAY]: [dayjs().add(-1, "d"),
        dayjs().add(-1, "d")],
    [TIME_RANGE_OPTION.LAST_7_DAYS]: [dayjs().add(-7, "d"),
        dayjs()],
    [TIME_RANGE_OPTION.LAST_30_DAYS]: [dayjs().add(-30, "d"),
        dayjs()],
    [TIME_RANGE_OPTION.MONTH_TO_DATE]: [dayjs().startOf("month"),
        dayjs()],

    // Custom option is just a placeholder for typing purposes, it's dayJs values should not
    // be used.
    [TIME_RANGE_OPTION.CUSTOM]: [dayjs(),
        dayjs()],
};


/**
 * Key names in enum `TIME_RANGE_OPTION`.
 */
const TIME_RANGE_OPTION_NAMES = Object.freeze(
    Object.values(TIME_RANGE_OPTION).filter((value) => "string" === typeof value)
);

/**
 * Validates dates provided by the range picker callback are non-null.
 *
 * @param dates
 * @return
 */
const isValidDateRange = (
    dates: [dayjs.Dayjs | null, dayjs.Dayjs | null] | null
): dates is [dayjs.Dayjs, dayjs.Dayjs] => {
    return null !== dates && null !== dates[0] && null !== dates[1];
};

export {
    isValidDateRange, TIME_RANGE_OPTION, TIME_RANGE_OPTION_NAMES, timeRangeOptionDayJsMap,
};
