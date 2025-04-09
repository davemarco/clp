import dayjs from "dayjs";
import {create} from "zustand";

import {
    TIME_RANGE_OPTION,
    timeRangeOptionDayJsMap,
} from "./SearchControls/TimeRangeInput/utils";


/**
 * Default values of the search context.
 */
const SEARCH_STATE_DEFAULT = Object.freeze({
    queryString: "",
    timeRange: timeRangeOptionDayJsMap[TIME_RANGE_OPTION.TODAY],
});

interface SearchState {
    queryString: string;
    timeRange: [dayjs.Dayjs, dayjs.Dayjs];
    updateQueryString: (query: string) => void;
    updateTimeRange: (range: [dayjs.Dayjs, dayjs.Dayjs]) => void;
}

const useSearchStore = create<SearchState>((set) => ({
    queryString: SEARCH_STATE_DEFAULT.queryString,
    timeRange: SEARCH_STATE_DEFAULT.timeRange,
    updateQueryString: (query) => {
        set({queryString: query});
    },
    updateTimeRange: (range) => {
        set({timeRange: range});
    },
}));

export {SEARCH_STATE_DEFAULT};
export default useSearchStore;
