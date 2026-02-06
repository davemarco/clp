#include "Output.hpp"

#include <string>

#include "../../../ColumnReader.hpp"
#include "../../../SchemaReader.hpp"
#include "../../../Utils.hpp"
#include "../../../search/OutputHandler.hpp"

namespace clp_s::gpu {
int emit_int_matches(
        SchemaReader& reader,
        std::vector<uint8_t> const& bitmap,
        search::OutputHandler& output_handler,
        std::string& error
) {
    std::string message;
    for (size_t i = 0; i < bitmap.size(); ++i) {
        if (0 == bitmap[i]) {
            continue;
        }
        if (false == reader.get_message_at(message, i)) {
            error = "failed to serialize message";
            return 1;
        }
        output_handler.write(message);
    }

    auto ecode = output_handler.flush();
    if (ErrorCode::ErrorCodeSuccess != ecode) {
        error = "failed to flush output handler";
        return 1;
    }
    return 0;
}
}  // namespace clp_s::gpu
