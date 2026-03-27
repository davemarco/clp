#include "SchemaReader.hpp"

#include <numeric>
#include <stack>
#include <string>

#include "archive_constants.hpp"
#include "BufferViewReader.hpp"
#include "Schema.hpp"

namespace clp_s {
void SchemaReader::append_column(BaseColumnReader* column_reader) {
    m_column_map[column_reader->get_id()] = column_reader;
    m_columns.push_back(column_reader);
}

void SchemaReader::append_unordered_column(BaseColumnReader* column_reader) {
    m_columns.push_back(column_reader);
}

void SchemaReader::add_structured_clp_string_reader(
        int32_t parent_id,
        Int64ColumnReader* logtype_reader,
        std::vector<Int64ColumnReader*> var_readers,
        std::shared_ptr<LogTypeDictionaryReader> log_dict,
        std::shared_ptr<VariableDictionaryReader> var_dict
) {
    if (nullptr == logtype_reader) {
        throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
    }
    m_structured_clp_string_reader_map.try_emplace(
            parent_id,
            logtype_reader,
            std::move(var_readers),
            std::move(log_dict),
            std::move(var_dict)
    );
}

void SchemaReader::mark_column_as_timestamp(BaseColumnReader* column_reader) {
    constexpr epochtime_t cNanosecondsInMillisecond{1000 * 1000LL};
    constexpr epochtime_t cMillisecondsInSecond{1000LL};
    m_timestamp_column = column_reader;
    if (m_timestamp_column->get_type() == NodeType::Timestamp) {
        m_get_timestamp = [this]() {
            return static_cast<TimestampColumnReader*>(m_timestamp_column)
                           ->get_encoded_time(m_cur_message)
                   / cNanosecondsInMillisecond;
        };
    } else if (m_timestamp_column->get_type() == NodeType::DeprecatedDateString) {
        m_get_timestamp = [this]() {
            return static_cast<DeprecatedDateStringColumnReader*>(m_timestamp_column)
                    ->get_encoded_time(m_cur_message);
            ;
        };
    } else if (m_timestamp_column->get_type() == NodeType::Integer) {
        m_get_timestamp = [this]() {
            return std::get<int64_t>(static_cast<Int64ColumnReader*>(m_timestamp_column)
                                             ->extract_value(m_cur_message));
        };
    } else if (m_timestamp_column->get_type() == NodeType::DeltaInteger) {
        m_get_timestamp = [this]() {
            return std::get<int64_t>(static_cast<DeltaEncodedInt64ColumnReader*>(m_timestamp_column)
                                             ->extract_value(m_cur_message));
        };
    } else if (m_timestamp_column->get_type() == NodeType::Float) {
        m_get_timestamp = [this]() {
            return static_cast<epochtime_t>(
                    std::get<double>(static_cast<FloatColumnReader*>(m_timestamp_column)
                                             ->extract_value(m_cur_message))
                    * cMillisecondsInSecond
            );
        };
    }
}

int64_t SchemaReader::get_next_log_event_idx() const {
    if (nullptr != m_log_event_idx_column) {
        return std::get<int64_t>(m_log_event_idx_column->extract_value(m_cur_message));
    }
    return 0;
}

void
SchemaReader::load(std::shared_ptr<char[]> stream_buffer, size_t offset, size_t uncompressed_size) {
    m_stream_buffer = stream_buffer;
    m_stream_buffer_offset = offset;
    m_stream_uncompressed_size = uncompressed_size;
    BufferViewReader buffer_reader{m_stream_buffer.get() + offset, uncompressed_size};
    for (auto& reader : m_columns) {
        reader->load(buffer_reader, m_num_messages);
    }
    if (buffer_reader.get_remaining_size() > 0) {
        throw OperationFailed(ErrorCodeCorrupt, __FILENAME__, __LINE__);
    }
}

void SchemaReader::generate_json_message(
        JsonSerializer& serializer,
        std::unordered_map<int32_t, StructuredClpStringReader>& sclp_map,
        size_t& sclp_index,
        uint64_t message_index
) {
    serializer.reset();
    serializer.begin_document();
    size_t column_id_index = 0;
    BaseColumnReader* column;
    JsonSerializer::Op op;
    while (serializer.get_next_op(op)) {
        switch (op) {
            case JsonSerializer::Op::BeginObject:
                serializer.begin_object();
                break;
            case JsonSerializer::Op::EndObject:
                serializer.end_object();
                break;
            case JsonSerializer::Op::BeginUnnamedObject:
                serializer.begin_document();
                break;
            case JsonSerializer::Op::BeginArray:
                serializer.begin_array();
                break;
            case JsonSerializer::Op::EndArray:
                serializer.end_array();
                break;
            case JsonSerializer::Op::BeginUnnamedArray:
                serializer.begin_array_document();
                break;
            case JsonSerializer::Op::AddIntField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value(
                        std::to_string(std::get<int64_t>(column->extract_value(message_index)))
                );
                break;
            case JsonSerializer::Op::AddIntValue:
                column = m_reordered_columns[column_id_index++];
                serializer.append_value(
                        std::to_string(std::get<int64_t>(column->extract_value(message_index)))
                );
                break;
            case JsonSerializer::Op::AddFloatField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value(
                        std::to_string(std::get<double>(column->extract_value(message_index)))
                );
                break;
            case JsonSerializer::Op::AddFloatValue:
                column = m_reordered_columns[column_id_index++];
                serializer.append_value(
                        std::to_string(std::get<double>(column->extract_value(message_index)))
                );
                break;
            case JsonSerializer::Op::AddFormattedFloatField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value_from_column(column, message_index);
                break;
            case JsonSerializer::Op::AddFormattedFloatValue:
                column = m_reordered_columns[column_id_index++];
                serializer.append_value_from_column(column, message_index);
                break;
            case JsonSerializer::Op::AddBoolField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value(
                        std::get<uint8_t>(column->extract_value(message_index)) != 0 ? "true"
                                                                                     : "false"
                );
                break;
            case JsonSerializer::Op::AddBoolValue:
                column = m_reordered_columns[column_id_index++];
                serializer.append_value(
                        std::get<uint8_t>(column->extract_value(message_index)) != 0 ? "true"
                                                                                     : "false"
                );
                break;
            case JsonSerializer::Op::AddStringField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value_from_column_with_quotes(column, message_index);
                break;
            case JsonSerializer::Op::AddStringValue:
                column = m_reordered_columns[column_id_index++];
                serializer.append_value_from_column_with_quotes(column, message_index);
                break;
            case JsonSerializer::Op::AddArrayField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value_from_column(column, message_index);
                break;
            case JsonSerializer::Op::AddNullField:
                serializer.append_key();
                serializer.append_value("null");
                break;
            case JsonSerializer::Op::AddNullValue:
                serializer.append_value("null");
                break;
            case JsonSerializer::Op::AddLiteralField:
                column = m_reordered_columns[column_id_index++];
                serializer.append_key(
                        m_global_schema_tree->get_node(column->get_id()).get_key_name()
                );
                serializer.append_value_from_column(column, message_index);
                break;
            case JsonSerializer::Op::AddStructuredClpStringField:
                serializer.append_key();
                serializer.append_escaped_string_value(
                        sclp_map.at(m_structured_clp_string_reader_ids[sclp_index++])
                                .decode(message_index)
                );
                break;
            case JsonSerializer::Op::AddStructuredClpStringValue:
                serializer.append_escaped_string_value(
                        sclp_map.at(m_structured_clp_string_reader_ids[sclp_index++])
                                .decode(message_index)
                );
                break;
        }
    }
    serializer.end_document();
}

auto SchemaReader::generate_json_string(uint64_t message_index) -> std::string {
    m_structured_clp_string_reader_index = 0;
    generate_json_message(
            m_json_serializer,
            m_structured_clp_string_reader_map,
            m_structured_clp_string_reader_index,
            message_index
    );
    return m_json_serializer.get_serialized_string();
}

bool SchemaReader::get_next_message(std::string& message) {
    if (m_cur_message >= m_num_messages) {
        return false;
    }

    if (false == m_serializer_initialized) {
        initialize_serializer();
    }
    message = generate_json_string(m_cur_message);

    if (message.back() != '\n') {
        message += '\n';
    }

    m_cur_message++;
    return true;
}

bool SchemaReader::get_next_message(std::string& message, FilterClass* filter) {
    while (m_cur_message < m_num_messages) {
        if (false == filter->filter(m_cur_message)) {
            m_cur_message++;
            continue;
        }

        if (m_should_marshal_records) {
            if (false == m_serializer_initialized) {
                initialize_serializer();
            }
            message = generate_json_string(m_cur_message);

            if (message.back() != '\n') {
                message += '\n';
            }
        }

        m_cur_message++;
        return true;
    }

    return false;
}

bool SchemaReader::get_next_message_with_metadata(
        std::string& message,
        epochtime_t& timestamp,
        int64_t& log_event_idx,
        FilterClass* filter
) {
    // TODO: If we already get max_num_results messages, we can skip messages
    // with the timestamp less than the smallest timestamp in the priority queue
    while (m_cur_message < m_num_messages) {
        if (false == filter->filter(m_cur_message)) {
            m_cur_message++;
            continue;
        }

        if (m_should_marshal_records) {
            if (false == m_serializer_initialized) {
                initialize_serializer();
            }
            message = generate_json_string(m_cur_message);

            if (message.back() != '\n') {
                message += '\n';
            }
        }

        timestamp = m_get_timestamp();
        log_event_idx = get_next_log_event_idx();

        m_cur_message++;
        return true;
    }

    return false;
}

/*** GPU integration start ***/
void SchemaReader::reset_read_state(uint64_t num_messages) {
    m_num_messages = num_messages;
    m_cur_message = 0;
    m_serializer_initialized = false;
}

void SchemaReader::serialize_indices_parallel(
        std::span<size_t const> indices,
        size_t num_threads,
        ThreadPool* thread_pool,
        std::vector<std::string>& per_thread_output
) {
    if (false == m_serializer_initialized) {
        initialize_serializer();
    }

    size_t const num_items = indices.size();
    if (0 == num_items) {
        return;
    }

    size_t const actual_threads = std::min(num_threads, num_items);
    per_thread_output.resize(actual_threads);

    if (actual_threads <= 1) {
        auto& output = per_thread_output[0];
        for (auto idx : indices) {
            auto msg = generate_json_string(idx);
            if (msg.empty() || msg.back() != '\n') {
                msg += '\n';
            }
            output += msg;
        }
        return;
    }

    // Each thread needs its own JsonSerializer and StructuredClpStringReader map.
    std::vector<JsonSerializer> thread_serializers(actual_threads, m_json_serializer);
    std::vector<std::unordered_map<int32_t, StructuredClpStringReader>> thread_sclp_maps;
    thread_sclp_maps.reserve(actual_threads);
    for (size_t t = 0; t < actual_threads; ++t) {
        thread_sclp_maps.emplace_back(m_structured_clp_string_reader_map);
    }

    size_t const items_per_thread = num_items / actual_threads;
    size_t const remainder = num_items % actual_threads;

    for (size_t t = 0; t < actual_threads; ++t) {
        size_t const start = t * items_per_thread + std::min(t, remainder);
        size_t const count = items_per_thread + (t < remainder ? 1 : 0);

        thread_pool->submit([this, t, start, count, &indices, &per_thread_output,
                             &thread_serializers, &thread_sclp_maps]() {
            auto& serializer = thread_serializers[t];
            auto& sclp_map = thread_sclp_maps[t];
            auto& output = per_thread_output[t];
            output.reserve(count * 256);

            for (size_t i = start; i < start + count; ++i) {
                size_t sclp_index = 0;
                generate_json_message(serializer, sclp_map, sclp_index, indices[i]);
                output += serializer.get_serialized_string();
                if (output.empty() || output.back() != '\n') {
                    output += '\n';
                }
            }
        });
    }
    thread_pool->wait_all();
}

void SchemaReader::serialize_range_parallel(
        size_t num_threads,
        ThreadPool* thread_pool,
        std::vector<std::string>& per_thread_output
) {
    std::vector<size_t> indices(m_num_messages);
    std::iota(indices.begin(), indices.end(), 0);
    serialize_indices_parallel(indices, num_threads, thread_pool, per_thread_output);
}

auto SchemaReader::serialize_message_at(uint64_t message_index) -> std::string {
    if (false == m_serializer_initialized) {
        initialize_serializer();
    }
    m_cur_message = message_index;
    auto message = generate_json_string(message_index);
    if (message.empty() || message.back() != '\n') {
        message += '\n';
    }
    return message;
}

void SchemaReader::serialize_bitmap_parallel(
        uint32_t const* bitmap,
        size_t num_rows,
        size_t num_threads,
        ThreadPool* thread_pool,
        std::vector<std::string>& per_thread_output
) {
    // Extract matching row indices from the packed bitmap.
    // For each word, __builtin_ctz finds the position of the lowest set bit,
    // then word &= word - 1 clears that bit so the next iteration finds the next one.
    // This skips zero words entirely and processes only set bits.
    size_t const num_words = (num_rows + 31) / 32;
    std::vector<size_t> match_indices;
    match_indices.reserve(num_rows / 10);  // Assume ~10% selectivity to reduce reallocations
    for (size_t w = 0; w < num_words; ++w) {
        uint32_t word = bitmap[w];
        while (word != 0) {
            int bit = __builtin_ctz(word);
            size_t row = w * 32 + static_cast<size_t>(bit);
            if (row < num_rows) {
                match_indices.push_back(row);
            }
            word &= word - 1;
        }
    }
    serialize_indices_parallel(match_indices, num_threads, thread_pool, per_thread_output);
}
/*** GPU integration end ***/

void SchemaReader::initialize_filter(FilterClass* filter) {
    filter->init(this, m_columns);
}

void SchemaReader::initialize_filter_with_column_map(FilterClass* filter) {
    filter->init(this, m_column_map);
}

void SchemaReader::generate_local_tree(int32_t global_id) {
    std::stack<int32_t> global_id_stack;
    global_id_stack.emplace(global_id);
    do {
        auto const& node = m_global_schema_tree->get_node(global_id_stack.top());
        int32_t parent_id = node.get_parent_id();

        auto it = m_global_id_to_local_id.find(parent_id);
        if (-1 != parent_id && it == m_global_id_to_local_id.end()) {
            global_id_stack.emplace(parent_id);
            continue;
        }

        int32_t local_id = m_local_schema_tree.add_node(
                parent_id == -1 ? -1 : m_global_id_to_local_id[parent_id],
                node.get_type(),
                node.get_key_name()
        );

        m_global_id_to_local_id[global_id_stack.top()] = local_id;
        m_local_id_to_global_id[local_id] = global_id_stack.top();
        global_id_stack.pop();
    } while (false == global_id_stack.empty());
}

void SchemaReader::mark_unordered_object(
        size_t column_reader_start,
        int32_t mst_subtree_root,
        std::span<int32_t> schema
) {
    m_global_id_to_unordered_object.emplace(
            mst_subtree_root,
            std::make_pair(column_reader_start, schema)
    );
}

int32_t SchemaReader::get_first_column_in_span(std::span<int32_t> schema) {
    for (int32_t column_id : schema) {
        if (false == Schema::schema_entry_is_unordered_object(column_id)) {
            return column_id;
        }
    }
    return -1;
}

void SchemaReader::find_intersection_and_fix_brackets(
        int32_t cur_root,
        int32_t next_root,
        std::vector<int32_t>& path_to_intersection
) {
    auto const* cur_node = &m_global_schema_tree->get_node(cur_root);
    auto const* next_node = &m_global_schema_tree->get_node(next_root);
    while (cur_node->get_parent_id() != next_node->get_parent_id()) {
        if (cur_node->get_depth() > next_node->get_depth()) {
            cur_root = cur_node->get_parent_id();
            cur_node = &m_global_schema_tree->get_node(cur_root);
            m_json_serializer.add_op(JsonSerializer::Op::EndObject);
        } else if (cur_node->get_depth() < next_node->get_depth()) {
            path_to_intersection.push_back(next_root);
            next_root = next_node->get_parent_id();
            next_node = &m_global_schema_tree->get_node(next_root);
        } else {
            cur_root = cur_node->get_parent_id();
            cur_node = &m_global_schema_tree->get_node(cur_root);
            m_json_serializer.add_op(JsonSerializer::Op::EndObject);
            path_to_intersection.push_back(next_root);
            next_root = next_node->get_parent_id();
            next_node = &m_global_schema_tree->get_node(next_root);
        }
    }

    // The loop above ends when the parent of next node and cur node matches. When these two nodes
    // have the same parent but are different nodes we need to close the last bracket for the
    // previous node, and add the first key for next node.
    if (cur_node != next_node) {
        m_json_serializer.add_op(JsonSerializer::Op::EndObject);
        path_to_intersection.push_back(next_node->get_id());
    }

    for (auto it = path_to_intersection.rbegin(); it != path_to_intersection.rend(); ++it) {
        auto const& node = m_global_schema_tree->get_node(*it);
        bool no_name = true;
        if (false == node.get_key_name().empty()) {
            m_json_serializer.add_special_key(node.get_key_name());
            no_name = false;
        }
        if (NodeType::Object == node.get_type()) {
            m_json_serializer.add_op(
                    no_name ? JsonSerializer::Op::BeginUnnamedObject
                            : JsonSerializer::Op::BeginObject
            );
        } else if (NodeType::StructuredArray == node.get_type()) {
            m_json_serializer.add_op(
                    no_name ? JsonSerializer::Op::BeginUnnamedArray : JsonSerializer::Op::BeginArray
            );
        }
    }
    path_to_intersection.clear();
}

size_t SchemaReader::generate_structured_array_template(
        int32_t array_root,
        size_t column_start,
        std::span<int32_t> schema
) {
    size_t column_idx = column_start;
    std::vector<int32_t> path_to_intersection;
    int32_t depth = m_global_schema_tree->get_node(array_root).get_depth();

    for (size_t i = 0; i < schema.size(); ++i) {
        int32_t global_column_id = schema[i];
        if (Schema::schema_entry_is_unordered_object(global_column_id)) {
            auto type = Schema::get_unordered_object_type(global_column_id);
            size_t length = Schema::get_unordered_object_length(global_column_id);
            auto sub_object_schema = schema.subspan(i + 1, length);
            if (NodeType::StructuredArray == type) {
                int32_t sub_array_root
                        = m_global_schema_tree->find_matching_subtree_root_in_subtree(
                                array_root,
                                get_first_column_in_span(sub_object_schema),
                                NodeType::StructuredArray
                        );
                m_json_serializer.add_op(JsonSerializer::Op::BeginUnnamedArray);
                column_idx = generate_structured_array_template(
                        sub_array_root,
                        column_idx,
                        sub_object_schema
                );
                m_json_serializer.add_op(JsonSerializer::Op::EndArray);
            } else if (NodeType::Object == type) {
                int32_t object_root = m_global_schema_tree->find_matching_subtree_root_in_subtree(
                        array_root,
                        get_first_column_in_span(sub_object_schema),
                        NodeType::Object
                );
                m_json_serializer.add_op(JsonSerializer::Op::BeginUnnamedObject);
                column_idx = generate_structured_object_template(
                        object_root,
                        column_idx,
                        sub_object_schema
                );
                m_json_serializer.add_op(JsonSerializer::Op::EndObject);
            } else if (NodeType::StructuredClpString == type) {
                int32_t clpstring_root
                        = m_global_schema_tree->find_matching_subtree_root_in_subtree(
                                array_root,
                                get_first_column_in_span(sub_object_schema),
                                NodeType::StructuredClpString
                        );
                m_json_serializer.add_op(JsonSerializer::Op::AddStructuredClpStringValue);
                column_idx = generate_structured_clpstring_template(
                        clpstring_root,
                        column_idx,
                        sub_object_schema
                );
            }
            i += length;
        } else {
            auto const& node = m_global_schema_tree->get_node(global_column_id);
            switch (node.get_type()) {
                case NodeType::Object: {
                    find_intersection_and_fix_brackets(
                            array_root,
                            global_column_id,
                            path_to_intersection
                    );
                    for (int j = 0; j < (node.get_depth() - depth); ++j) {
                        m_json_serializer.add_op(JsonSerializer::Op::EndObject);
                    }
                    break;
                }
                case NodeType::StructuredArray: {
                    m_json_serializer.add_op(JsonSerializer::Op::BeginUnnamedArray);
                    m_json_serializer.add_op(JsonSerializer::Op::EndArray);
                    break;
                }
                case NodeType::DeltaInteger:
                case NodeType::Integer: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddIntValue);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::Float: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddFloatValue);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::FormattedFloat:
                case NodeType::DictionaryFloat: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddFormattedFloatValue);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::Boolean: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddBoolValue);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::ClpString:
                case NodeType::VarString: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddStringValue);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::NullValue: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddNullValue);
                    break;
                }
                case NodeType::DeprecatedDateString:
                case NodeType::UnstructuredArray:
                case NodeType::Metadata:
                case NodeType::Timestamp:
                case NodeType::Unknown:
                    break;
            }
        }
    }
    return column_idx;
}

size_t SchemaReader::generate_structured_object_template(
        int32_t object_root,
        size_t column_start,
        std::span<int32_t> schema
) {
    int32_t root = object_root;
    size_t column_idx = column_start;
    std::vector<int32_t> path_to_intersection;

    for (size_t i = 0; i < schema.size(); ++i) {
        int32_t global_column_id = schema[i];
        if (Schema::schema_entry_is_unordered_object(global_column_id)) {
            auto unordered_type = Schema::get_unordered_object_type(global_column_id);
            size_t sub_length = Schema::get_unordered_object_length(global_column_id);
            auto sub_schema = schema.subspan(i + 1, sub_length);
            if (NodeType::StructuredClpString == unordered_type) {
                int32_t clpstring_root
                        = m_global_schema_tree->find_matching_subtree_root_in_subtree(
                                object_root,
                                get_first_column_in_span(sub_schema),
                                NodeType::StructuredClpString
                        );
                find_intersection_and_fix_brackets(
                        root,
                        m_global_schema_tree->get_node(clpstring_root).get_parent_id(),
                        path_to_intersection
                );
                m_json_serializer.add_op(JsonSerializer::Op::AddStructuredClpStringField);
                m_json_serializer.add_special_key(
                        m_global_schema_tree->get_node(clpstring_root).get_key_name()
                );
                column_idx = generate_structured_clpstring_template(
                        clpstring_root,
                        column_idx,
                        sub_schema
                );
                root = m_global_schema_tree->get_node(clpstring_root).get_parent_id();
            } else {
                // Arrays (and potentially other nested unordered types) inside structured objects
                int32_t array_root
                        = m_global_schema_tree->find_matching_subtree_root_in_subtree(
                                object_root,
                                get_first_column_in_span(sub_schema),
                                NodeType::StructuredArray
                        );

                find_intersection_and_fix_brackets(root, array_root, path_to_intersection);
                column_idx = generate_structured_array_template(
                        array_root,
                        column_idx,
                        sub_schema
                );
                m_json_serializer.add_op(JsonSerializer::Op::EndArray);
                // root is parent of the array object since we close the array bracket above
                auto const& node = m_global_schema_tree->get_node(array_root);
                root = node.get_parent_id();
            }
            i += sub_length;
        } else {
            auto const& node = m_global_schema_tree->get_node(global_column_id);
            int32_t next_root = node.get_parent_id();
            find_intersection_and_fix_brackets(root, next_root, path_to_intersection);
            root = next_root;
            switch (node.get_type()) {
                case NodeType::Object: {
                    m_json_serializer.add_op(JsonSerializer::Op::BeginObject);
                    m_json_serializer.add_special_key(node.get_key_name());
                    m_json_serializer.add_op(JsonSerializer::Op::EndObject);
                    break;
                }
                case NodeType::StructuredArray: {
                    m_json_serializer.add_op(JsonSerializer::Op::BeginArray);
                    m_json_serializer.add_special_key(node.get_key_name());
                    m_json_serializer.add_op(JsonSerializer::Op::EndArray);
                    break;
                }
                case NodeType::DeltaInteger:
                case NodeType::Integer: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddIntField);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::Float: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddFloatField);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::FormattedFloat:
                case NodeType::DictionaryFloat: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddFormattedFloatField);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::Boolean: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddBoolField);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::ClpString:
                case NodeType::VarString: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddStringField);
                    m_reordered_columns.push_back(m_columns[column_idx++]);
                    break;
                }
                case NodeType::NullValue: {
                    m_json_serializer.add_op(JsonSerializer::Op::AddNullField);
                    m_json_serializer.add_special_key(node.get_key_name());
                    break;
                }
                case NodeType::DeprecatedDateString:
                case NodeType::UnstructuredArray:
                case NodeType::Metadata:
                case NodeType::Timestamp:
                case NodeType::Unknown:
                    break;
            }
        }
    }
    find_intersection_and_fix_brackets(root, object_root, path_to_intersection);
    return column_idx;
}

void SchemaReader::initialize_serializer() {
    if (m_serializer_initialized) {
        return;
    }

    m_serializer_initialized = true;

    for (int32_t global_column_id : m_ordered_schema) {
        if (m_projection->matches_node(global_column_id)) {
            generate_local_tree(global_column_id);
        }
    }

    for (auto it = m_global_id_to_unordered_object.begin();
         it != m_global_id_to_unordered_object.end();
         ++it)
    {
        if (m_projection->matches_node(it->first)) {
            generate_local_tree(it->first);
        }
    }

    // TODO: this code will have to change once we allow mixing log lines parsed by different
    // parsers and if we add support for serializing auto-generated keys in regular JSON.
    if (auto subtree_root = m_local_schema_tree.get_object_subtree_node_id_for_namespace(
                constants::cDefaultNamespace
        );
        -1 != subtree_root)
    {
        generate_json_template(subtree_root);
    }
}

size_t SchemaReader::generate_structured_clpstring_template(
        int32_t clpstring_root,
        size_t column_start,
        std::span<int32_t> schema
) {
    m_structured_clp_string_reader_ids.push_back(clpstring_root);
    size_t column_idx = column_start;
    for (int32_t entry : schema) {
        if (false == Schema::schema_entry_is_unordered_object(entry)) {
            column_idx++;
        }
    }
    return column_idx;
}

void SchemaReader::generate_json_template(int32_t id) {
    auto const& node = m_local_schema_tree.get_node(id);
    auto const& children_ids = node.get_children_ids();

    for (int32_t child_id : children_ids) {
        int32_t child_global_id = m_local_id_to_global_id[child_id];
        auto const& child_node = m_local_schema_tree.get_node(child_id);
        auto key = child_node.get_key_name();
        switch (child_node.get_type()) {
            case NodeType::Object: {
                m_json_serializer.add_op(JsonSerializer::Op::BeginObject);
                m_json_serializer.add_special_key(key);
                generate_json_template(child_id);
                m_json_serializer.add_op(JsonSerializer::Op::EndObject);
                break;
            }
            case NodeType::UnstructuredArray: {
                m_json_serializer.add_op(JsonSerializer::Op::AddArrayField);
                m_reordered_columns.push_back(m_column_map[child_global_id]);
                break;
            }
            case NodeType::StructuredArray: {
                m_json_serializer.add_op(JsonSerializer::Op::BeginArray);
                m_json_serializer.add_special_key(key);
                auto structured_it = m_global_id_to_unordered_object.find(child_global_id);
                if (m_global_id_to_unordered_object.end() != structured_it) {
                    size_t column_start = structured_it->second.first;
                    std::span<int32_t> structured_schema = structured_it->second.second;
                    generate_structured_array_template(
                            child_global_id,
                            column_start,
                            structured_schema
                    );
                }
                m_json_serializer.add_op(JsonSerializer::Op::EndArray);
                break;
            }
            case NodeType::StructuredClpString: {
                m_json_serializer.add_op(JsonSerializer::Op::AddStructuredClpStringField);
                m_json_serializer.add_special_key(key);
                auto structured_it = m_global_id_to_unordered_object.find(child_global_id);
                if (m_global_id_to_unordered_object.end() != structured_it) {
                    generate_structured_clpstring_template(
                            child_global_id,
                            structured_it->second.first,
                            structured_it->second.second
                    );
                }
                break;
            }
            case NodeType::DeltaInteger:
            case NodeType::Integer: {
                m_json_serializer.add_op(JsonSerializer::Op::AddIntField);
                m_reordered_columns.push_back(m_column_map[child_global_id]);
                break;
            }
            case NodeType::Float: {
                m_json_serializer.add_op(JsonSerializer::Op::AddFloatField);
                m_reordered_columns.push_back(m_column_map[child_global_id]);
                break;
            }
            case NodeType::FormattedFloat:
            case NodeType::DictionaryFloat: {
                m_json_serializer.add_op(JsonSerializer::Op::AddFormattedFloatField);
                m_reordered_columns.push_back(m_column_map[child_global_id]);
                break;
            }
            case NodeType::Boolean: {
                m_json_serializer.add_op(JsonSerializer::Op::AddBoolField);
                m_reordered_columns.push_back(m_column_map[child_global_id]);
                break;
            }
            case NodeType::ClpString:
            case NodeType::VarString:
            case NodeType::DeprecatedDateString: {
                m_json_serializer.add_op(JsonSerializer::Op::AddStringField);
                m_reordered_columns.push_back(m_column_map[child_global_id]);
                break;
            }
            case NodeType::Timestamp: {
                m_json_serializer.add_op(JsonSerializer::Op::AddLiteralField);
                m_reordered_columns.emplace_back(m_column_map.at(child_global_id));
                break;
            }
            case NodeType::NullValue: {
                m_json_serializer.add_op(JsonSerializer::Op::AddNullField);
                m_json_serializer.add_special_key(key);
                break;
            }
            case NodeType::Metadata:
            case NodeType::Unknown:
                break;
        }
    }
}
}  // namespace clp_s
