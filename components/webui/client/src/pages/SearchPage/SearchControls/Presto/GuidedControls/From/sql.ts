import {querySynchronous, PrestoResultsSchema} from "../../../../../../api/presto-search";

/**
 * SQL query to get all table names.
 */
const GET_TABLES_SQL = "SHOW TABLES";

/**
 * Fetches table names from Presto using SHOW TABLES.
 *
 * @return
 */
const fetchPrestoTableNames = async (): Promise<string[]> => {
    const resp = await querySynchronous({queryString: GET_TABLES_SQL});
    const result: PrestoResultsSchema = resp.data;
    if (!result.results || result.results.length === 0) return [];
    return result.results.map((row) => {
        if (typeof row === "object" && row !== null && "Table" in row) {
            return row["Table"] as string;
        }
        return "";
    });
};

export {GET_TABLES_SQL, fetchPrestoTableNames};
