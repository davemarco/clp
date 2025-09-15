import {
    Static,
    Type,
} from "@sinclair/typebox";

import {StringSchema} from "./common.js";


/**
 * Schema for request to create a new Presto query job.
 */
const PrestoQueryJobCreationSchema = Type.Object({
    queryString: StringSchema,
});

/**
 * Schema to identify a Presto query job.
 */
const PrestoQueryJobSchema = Type.Object({
    searchJobId: StringSchema,
});

/**
 * Schema for Presto column metadata.
 */
const PrestoSynchronousColumnSchema = Type.Object({
    name: Type.String(),
    type: Type.String(),
    typeSignature: Type.Any(),
});

/**
 * Schema for Presto synchronous query results.
 */
const PrestoQuerySynchronousResultsSchema = Type.Object({
    columns: Type.Array(PrestoSynchronousColumnSchema),
    data: Type.Array(Type.Any()),
});

type PrestoQueryJobCreation = Static<typeof PrestoQueryJobCreationSchema>;

type PrestoQueryJob = Static<typeof PrestoQueryJobSchema>;

type PrestoQuerySynchronousResults = Static<typeof PrestoQuerySynchronousResultsSchema>;


export {
    PrestoQueryJobCreationSchema,
    PrestoQueryJobSchema,
    PrestoQuerySynchronousResultsSchema,
};
export type {
    PrestoQueryJob,
    PrestoQueryJobCreation,
    PrestoQuerySynchronousResults,
};
