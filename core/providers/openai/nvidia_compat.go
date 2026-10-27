package openai

import (
	"encoding/json"
	"fmt"
	"maps"
	"strings"

	"github.com/maximhq/bifrost/core/schemas"
)

// NVIDIA NIM exposes an OpenAI-compatible surface but rejects several parameters
// and input item shapes that the real APIs accept. These helpers keep the
// OpenAI provider wire-compatible with NIM without affecting other upstreams.

const nvidiaBaseURLFragment = "integrate.api.nvidia.com"

func isNvidiaUpstream(baseURL string) bool {
	return strings.Contains(strings.ToLower(baseURL), nvidiaBaseURLFragment)
}

// sanitizeNvidiaChatRequest drops parameters NIM rejects with
// "Unsupported parameter(s)".
func sanitizeNvidiaChatRequest(request *schemas.BifrostChatRequest) {
	if request == nil || request.Params == nil || request.Params.ExtraParams == nil {
		return
	}
	if _, ok := request.Params.ExtraParams["enable_thinking"]; !ok {
		return
	}
	params := *request.Params
	params.ExtraParams = maps.Clone(request.Params.ExtraParams)
	delete(params.ExtraParams, "enable_thinking")
	request.Params = &params
}

// Input item types NIM accepts unchanged.
var nvidiaSupportedInputTypes = map[string]struct{}{
	"reasoning":             {},
	"item_reference":        {},
	"image_generation_call": {},
}

// normalizeNvidiaResponsesInput rewrites Responses API input items that NIM
// cannot deserialize. NIM only accepts a strict subset of item types; built-in
// tool calls, non-string tool payloads and tool roles make the whole request
// fail with "data did not match any variant of untagged enum InputParam".
// Unknown item types are degraded to text messages so they cannot break the
// request, matching the compatibility layer new-api applies to the same
// upstream.
func normalizeNvidiaResponsesInput(input []schemas.ResponsesMessage) []schemas.ResponsesMessage {
	if len(input) == 0 {
		return input
	}
	out := make([]schemas.ResponsesMessage, 0, len(input))
	changed := false
	for i := range input {
		raw, err := json.Marshal(input[i])
		if err != nil {
			out = append(out, input[i])
			continue
		}
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			out = append(out, input[i])
			continue
		}
		converted, didChange := nvidiaNormalizeInputItem(item, i)
		if !didChange {
			out = append(out, input[i])
			continue
		}
		encoded, err := json.Marshal(converted)
		if err != nil {
			out = append(out, input[i])
			continue
		}
		var message schemas.ResponsesMessage
		if err := json.Unmarshal(encoded, &message); err != nil {
			out = append(out, input[i])
			continue
		}
		out = append(out, message)
		changed = true
	}
	if !changed {
		return input
	}
	return out
}

func nvidiaNormalizeInputItem(item map[string]any, index int) (map[string]any, bool) {
	itemType, _ := item["type"].(string)
	switch itemType {
	case "", "message":
		role, _ := item["role"].(string)
		if role == "tool" || role == "function" {
			item["role"] = "user"
			return item, true
		}
		return item, false
	case "function_call":
		arguments, ok := item["arguments"]
		if !ok {
			return item, false
		}
		if _, isString := arguments.(string); isString {
			return item, false
		}
		encoded, err := json.Marshal(arguments)
		if err != nil {
			return item, false
		}
		item["arguments"] = string(encoded)
		return item, true
	case "function_call_output":
		output, ok := item["output"]
		if !ok {
			return item, false
		}
		if _, isString := output.(string); isString {
			return item, false
		}
		encoded, err := json.Marshal(output)
		if err != nil {
			return item, false
		}
		item["output"] = string(encoded)
		return item, true
	case "local_shell_call", "custom_tool_call", "computer_call",
		"code_interpreter_call", "mcp_call", "web_search_call",
		"web_fetch_call", "file_search_call", "shell_call",
		"apply_patch_call", "tool_search_call":
		return nvidiaFunctionCallItem(item, itemType, index), true
	case "local_shell_call_output", "custom_tool_call_output", "computer_call_output",
		"shell_call_output", "apply_patch_call_output", "tool_search_output":
		return nvidiaFunctionCallOutputItem(item, index), true
	default:
		if _, ok := nvidiaSupportedInputTypes[itemType]; ok {
			return item, false
		}
		return nvidiaTextMessageItem(item, itemType), true
	}
}

func nvidiaFunctionCallItem(item map[string]any, itemType string, index int) map[string]any {
	return map[string]any{
		"type":      "function_call",
		"name":      nvidiaFunctionName(item, itemType),
		"arguments": nvidiaCallArguments(item),
		"call_id":   nvidiaCallID(item, index),
	}
}

func nvidiaFunctionCallOutputItem(item map[string]any, index int) map[string]any {
	output := ""
	switch value := item["output"].(type) {
	case string:
		output = value
	case nil:
		output = ""
	default:
		if encoded, err := json.Marshal(value); err == nil {
			output = string(encoded)
		}
	}
	return map[string]any{
		"type":    "function_call_output",
		"call_id": nvidiaCallID(item, index),
		"output":  output,
	}
}

func nvidiaFunctionName(item map[string]any, itemType string) string {
	switch itemType {
	case "local_shell_call":
		return "local_shell"
	case "computer_call":
		return "computer"
	case "code_interpreter_call":
		return "code_interpreter"
	case "web_search_call":
		return "web_search"
	case "web_fetch_call":
		return "web_fetch"
	case "file_search_call":
		return "file_search"
	case "shell_call":
		return "shell"
	case "apply_patch_call":
		return "apply_patch"
	case "tool_search_call":
		return "tool_search"
	case "mcp_call", "custom_tool_call":
		if name, ok := item["name"].(string); ok && name != "" {
			return name
		}
		return strings.TrimSuffix(itemType, "_call")
	}
	return strings.TrimSuffix(itemType, "_call")
}

func nvidiaCallArguments(item map[string]any) string {
	for _, key := range []string{"arguments", "input", "action", "code"} {
		value, ok := item[key]
		if !ok {
			continue
		}
		if text, ok := value.(string); ok {
			return text
		}
		if encoded, err := json.Marshal(value); err == nil {
			return string(encoded)
		}
	}
	withoutType := make(map[string]any, len(item))
	for key, value := range item {
		if key != "type" {
			withoutType[key] = value
		}
	}
	if encoded, err := json.Marshal(withoutType); err == nil {
		return string(encoded)
	}
	return "{}"
}

func nvidiaCallID(item map[string]any, index int) string {
	if callID, ok := item["call_id"].(string); ok && callID != "" {
		return callID
	}
	if id, ok := item["id"].(string); ok && id != "" {
		return id
	}
	return fmt.Sprintf("nvidia_call_%d", index)
}

func nvidiaTextMessageItem(item map[string]any, itemType string) map[string]any {
	role := "user"
	contentType := "input_text"
	if strings.HasSuffix(itemType, "_call") {
		role = "assistant"
		contentType = "output_text"
	}
	encoded, err := json.Marshal(item)
	if err != nil {
		encoded = []byte("{}")
	}
	return map[string]any{
		"type":    "message",
		"role":    role,
		"content": []map[string]any{{"type": contentType, "text": string(encoded)}},
	}
}
