import {
    type PrestoQueryJob,
    type PrestoQueryJobCreation,
} from "@webui/common/schemas/presto-search";
import axios, {AxiosResponse} from "axios";


/**
 * Sends post request to server to submit presto query.
 *
 * @param payload
 * @return
 */
const submitQuery = async (
    payload: PrestoQueryJobCreation
): Promise<AxiosResponse<PrestoQueryJob>> => {
    console.log("Submitting query:", JSON.stringify(payload));

    return axios.post<PrestoQueryJob>("/api/presto-search/query", payload);
};


/**
 * Sends post request to server to cancel presto query.
 *
 * @param payload
 * @return
 */
const cancelQuery = async (
    payload: PrestoQueryJob
): Promise<AxiosResponse<null>> => {
    console.log("Cancelling query:", JSON.stringify(payload));

    return axios.post("/api/presto-search/cancel", payload);
};


/**
 * Sends delete request to server to clear presto query results.
 *
 * @param payload
 * @return
 */
const clearQueryResults = (payload: PrestoQueryJob): Promise<AxiosResponse<null>> => {
    console.log("Clearing query:", JSON.stringify(payload));

    return axios.delete("/api/presto-search/results", {data: payload});
};

// PrestoResultsSchema type for synchronous results
export type PrestoResultsSchema = {
    results: any[];
    columns: Array<{
        name: string;
        type: string;
        typeSignature: any;
    }>;
};

/**
 * Sends post request to server for synchronous presto query.
 *
 * @param payload
 * @return
 */
const querySynchronous = async (
    payload: PrestoQueryJobCreationSchema
): Promise<AxiosResponse<PrestoResultsSchema>> => {
    return axios.post<PrestoResultsSchema>("/api/presto-search/query-synchronous", payload);
};

export type {
    PrestoQueryJobCreationSchema,
    PrestoQueryJobSchema,
};

export {
    cancelQuery,
    clearQueryResults,
    submitQuery,
    querySynchronous,
};
