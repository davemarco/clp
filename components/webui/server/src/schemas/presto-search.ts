import {Type} from "@sinclair/typebox";

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
const PrestoColumnSchema = Type.Object({
    name: Type.String(),
    type: Type.String(),
    typeSignature: Type.Any(),
});

/**
 * Schema for Presto synchronous query results.
 */
const PrestoSynchronousSchema = Type.Object({
    results: Type.Array(Type.Any()),
    columns: Type.Array(PrestoColumnSchema),
});

export {
    PrestoQueryJobCreationSchema,
    PrestoQueryJobSchema,
    PrestoSynchronousSchema,
};
