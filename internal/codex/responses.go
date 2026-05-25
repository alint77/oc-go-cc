package codex

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"oc-go-cc/internal/config"
	"oc-go-cc/pkg/types"
)

// ResponsesRequest is the OpenAI Responses API request format used by Codex.
type ResponsesRequest struct {
	Model        string          `json:"model"`
	Input        json.RawMessage `json:"input"`
	Instructions string          `json:"instructions"`
	Store        bool            `json:"store"`
	Stream       bool            `json:"stream"`
	Temperature  *float64        `json:"temperature,omitempty"`
	TopP         *float64        `json:"top_p,omitempty"`
	Tools        json.RawMessage `json:"tools,omitempty"`
	ToolChoice   interface{}     `json:"tool_choice,omitempty"`
	Reasoning    json.RawMessage `json:"reasoning,omitempty"`
}

// ResponsesEvent represents a single SSE event from the Responses API.
type ResponsesEvent struct {
	Event string          `json:"event,omitempty"`
	Type  string          `json:"type"`
	Data  json.RawMessage `json:"data,omitempty"`
}

// ToOpenAIResponse converts an Anthropic request to the Codex Responses API format.
func ToOpenAIResponse(anthropicReq *types.MessageRequest, model config.ModelConfig) ([]byte, error) {
	instructions := anthropicReq.SystemText()

	input, err := buildInput(anthropicReq.Messages)
	if err != nil {
		return nil, fmt.Errorf("build input: %w", err)
	}

	req := ResponsesRequest{
		Model:        model.ModelID,
		Input:        input,
		Instructions: instructions,
		Store:        false,
		Stream:       true,
	}

	// Map Claude Code's thinking budget_tokens to Codex reasoning effort.
	// Claude Code sends thinking config when user sets effort per-model.
	if anthropicReq.Thinking != nil && anthropicReq.Thinking.Type == "enabled" {
		effort := mapEffortFromThinkingBudget(anthropicReq.Thinking.BudgetTokens)
		if effort != "" {
			reasoning := map[string]interface{}{
				"effort": effort,
			}
			r, _ := json.Marshal(reasoning)
			req.Reasoning = r
		}
	}

	// Codex endpoint rejects temperature/top_p - skip them

	if len(anthropicReq.Tools) > 0 {
		tools, err := buildTools(anthropicReq.Tools)
		if err != nil {
			return nil, fmt.Errorf("build tools: %w", err)
		}
		req.Tools = tools
	}

	return json.Marshal(req)
}

func buildInput(messages []types.Message) (json.RawMessage, error) {
	var input []map[string]interface{}
	for _, msg := range messages {
		blocks := msg.ContentBlocks()
		role := msg.Role

		switch role {
		case "user":
			// Extract text and tool results from user messages
			var textParts []string
			for _, b := range blocks {
				switch b.Type {
				case "text":
					textParts = append(textParts, b.Text)
				case "tool_result":
					// Add tool result as function_call_output (Responses API format)
					toolContent := b.TextContent()
					toolUseID := b.ToolUseID
					tcr := map[string]interface{}{
						"type":   "function_call_output",
						"output": toolContent,
					}
					if toolUseID != "" {
						tcr["call_id"] = toolUseID
					}
					input = append(input, tcr)
				}
			}
			if len(textParts) > 0 {
				input = append(input, map[string]interface{}{
					"role":    "user",
					"content": strings.Join(textParts, ""),
				})
			}
		case "assistant":
			textContent := extractText(blocks)
			// Add assistant text content FIRST
			item := map[string]interface{}{
				"role": "assistant",
			}
			if textContent != "" {
				item["content"] = textContent
			}
			input = append(input, item)
			// THEN add tool calls as separate items
			for _, b := range blocks {
				if b.Type == "tool_use" {
					input = append(input, map[string]interface{}{
						"type":      "function_call",
						"call_id":   b.ID,
						"name":      b.Name,
						"arguments": string(b.Input),
					})
				}
			}
		}
	}
	if len(input) == 0 {
		input = []map[string]interface{}{}
	}
	data, _ := json.Marshal(input)
	return data, nil
}

func extractText(blocks []types.ContentBlock) string {
	var parts []string
	for _, b := range blocks {
		switch b.Type {
		case "text":
			parts = append(parts, b.Text)
		case "thinking":
			// Skip thinking blocks
		case "tool_use":
			// Skip tool use in text extraction
		case "tool_result":
			parts = append(parts, b.TextContent())
		case "image":
			parts = append(parts, "[Image]")
		}
	}
	return strings.Join(parts, "")
}

type toolCallInfo struct {
	Name string
}

func extractToolCalls(blocks []types.ContentBlock) []toolCallInfo {
	var calls []toolCallInfo
	for _, b := range blocks {
		if b.Type == "tool_use" {
			calls = append(calls, toolCallInfo{Name: b.Name})
		}
	}
	return calls
}

// mapEffortFromThinkingBudget maps Claude Code's thinking budget_tokens
// to Codex reasoning_effort level. Higher budget = more reasoning depth.
// Claude Code maps effort levels to approximate budget_tokens:
//
//	none     → no thinking
//	low      → ~1024
//	medium   → ~4096
//	high     → ~8192
//	xhigh    → ~16384
//	max      → ~32000
func mapEffortFromThinkingBudget(budget int) string {
	switch {
	case budget >= 24000:
		return "xhigh"
	case budget >= 12000:
		return "high"
	case budget >= 3000:
		return "medium"
	case budget >= 1:
		return "low"
	default:
		return ""
	}
}

func buildTools(anthropicTools []types.Tool) (json.RawMessage, error) {
	var tools []map[string]interface{}
	for _, t := range anthropicTools {
		tools = append(tools, map[string]interface{}{
			"type":        "function",
			"name":        t.Name,
			"description": t.Description,
			"parameters":  t.InputSchema,
		})
	}
	return json.Marshal(tools)
}

// ParseSSELine parses a single SSE line and returns event/data pairs.
// Returns event name, data string, and whether this is a valid data line.
func ParseSSELine(line string) (string, string, bool) {
	if strings.HasPrefix(line, "event: ") {
		return strings.TrimPrefix(line, "event: "), "", true
	}
	if strings.HasPrefix(line, "data: ") {
		return "", strings.TrimPrefix(line, "data: "), true
	}
	return "", "", false
}

// BuildAnthropicChunk converts a Responses SSE event into an Anthropic-compatible
// JSON byte slice for streaming, plus a boolean indicating if this is a final event.
func BuildAnthropicChunk(eventType string, data []byte) (chunk []byte, isFinal bool, err error) {
	switch eventType {
	case "response.output_text.delta":
		var d struct {
			Delta string `json:"delta"`
		}
		if err := json.Unmarshal(data, &d); err != nil {
			return nil, false, err
		}
		b, err := json.Marshal(map[string]interface{}{
			"type": "content_block_delta",
			"delta": map[string]interface{}{
				"type": "text_delta",
				"text": d.Delta,
			},
		})
		return b, false, err

	case "response.output_text.done":
		b, err := json.Marshal(map[string]interface{}{
			"type": "content_block_stop",
		})
		return b, false, err

	case "response.content_part.added":
		var d struct {
			Part struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"part"`
		}
		if err := json.Unmarshal(data, &d); err != nil {
			return nil, false, err
		}
		if d.Part.Type == "output_text" {
			b, err := json.Marshal(map[string]interface{}{
				"type":  "content_block_start",
				"index": 0,
				"content_block": map[string]interface{}{
					"type": "text",
					"text": d.Part.Text,
				},
			})
			return b, false, err
		}
		return nil, false, nil

	case "response.output_item.added":
		var d struct {
			Item struct {
				Type string `json:"type"`
			} `json:"item"`
		}
		if err := json.Unmarshal(data, &d); err != nil {
			return nil, false, err
		}
		if d.Item.Type == "message" {
			b, err := json.Marshal(map[string]interface{}{
				"type": "content_block_start",
				"content_block": map[string]interface{}{
					"type": "text",
					"text": "",
				},
			})
			return b, false, err
		}
		return nil, false, nil

	case "response.completed":
		b, err := json.Marshal(map[string]interface{}{
			"type": "message_delta",
			"delta": map[string]interface{}{
				"stop_reason":   "end_turn",
				"stop_sequence": nil,
			},
		})
		return b, false, err

	case "response.created", "response.in_progress":
		return nil, false, nil
	}
	return nil, false, nil
}

// BuildCompletionResponseFromSSE parses all collected SSE data and builds
// a complete Anthropic MessageResponse for non-streaming use.
func BuildCompletionResponseFromSSE(events []SSEEvent, originalModel string) ([]byte, error) {
	var fullText strings.Builder
	var toolCalls []map[string]interface{}

	for _, ev := range events {
		switch ev.EventType {
		case "response.output_text.delta":
			var d struct{ Delta string `json:"delta"` }
			json.Unmarshal(ev.Data, &d)
			fullText.WriteString(d.Delta)
		case "response.output_item.done":
			var d struct {
				Item struct {
					Type    string `json:"type"`
					Name    string `json:"name"`
					Arguments string `json:"arguments"`
					CallID  string `json:"call_id"`
				} `json:"item"`
			}
			json.Unmarshal(ev.Data, &d)
			if d.Item.Type == "function_call" {
		toolCalls = append(toolCalls, map[string]interface{}{
					"type": "tool_use",
					"id":   d.Item.CallID,
					"name": d.Item.Name,
					"input": json.RawMessage(d.Item.Arguments),
				})
			}
		}
	}

	var contentBlocks []map[string]interface{}
	if fullText.Len() > 0 {
		contentBlocks = append(contentBlocks, map[string]interface{}{
			"type": "text",
			"text": fullText.String(),
		})
	}
	for _, tc := range toolCalls {
		contentBlocks = append(contentBlocks, tc)
	}
	if len(contentBlocks) == 0 {
		contentBlocks = append(contentBlocks, map[string]interface{}{
			"type": "text",
			"text": "",
		})
	}

	stopReason := "end_turn"
	resp := map[string]interface{}{
		"id":      fmt.Sprintf("msg_%x", time.Now().UnixNano()),
		"type":    "message",
		"role":    "assistant",
		"content": contentBlocks,
		"model":   originalModel,
		"stop_reason":   stopReason,
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  0,
			"output_tokens": 0,
		},
	}
	return json.Marshal(resp)
}

// SSEEvent holds a parsed SSE event for non-streaming consumption.
type SSEEvent struct {
	EventType string
	Data      json.RawMessage
}
