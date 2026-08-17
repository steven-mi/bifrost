package huggingface

import (
	"encoding/json"
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToHuggingFaceChatCompletionRequest_ResponseFormat(t *testing.T) {
	makeReq := func(rf *schemas.ChatResponseFormat) *schemas.BifrostChatRequest {
		return &schemas.BifrostChatRequest{
			Model: "test-model",
			Input: []schemas.ChatMessage{{Role: schemas.ChatMessageRoleUser, Content: &schemas.ChatMessageContent{ContentStr: schemas.Ptr("hello")}}},
			Params: &schemas.ChatParameters{
				ResponseFormat: rf,
			},
		}
	}

	tests := []struct {
		name           string
		responseFormat *schemas.ChatResponseFormat
		wantErr        bool
		validate       func(t *testing.T, result *HuggingFaceChatRequest)
	}{
		{
			name:           "nil_response_format",
			responseFormat: nil,
			validate: func(t *testing.T, result *HuggingFaceChatRequest) {
				assert.Nil(t, result.ResponseFormat)
			},
		},
		{
			name:           "type_only",
			responseFormat: schemas.NewChatResponseFormatFromMap(map[string]interface{}{"type": "json_object"}),
			validate: func(t *testing.T, result *HuggingFaceChatRequest) {
				require.NotNil(t, result.ResponseFormat)
				assert.Equal(t, "json_object", result.ResponseFormat.Type)
				assert.Nil(t, result.ResponseFormat.JSONSchema)
			},
		},
		{
			name: "json_schema",
			responseFormat: schemas.NewChatResponseFormatFromMap(map[string]interface{}{
				"type": "json_schema",
				"json_schema": map[string]interface{}{
					"name":        "my_schema",
					"description": "A test schema",
					"strict":      true,
					"schema": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"answer": map[string]interface{}{"type": "string"},
						},
					},
				},
			}),
			validate: func(t *testing.T, result *HuggingFaceChatRequest) {
				require.NotNil(t, result.ResponseFormat)
				assert.Equal(t, "json_schema", result.ResponseFormat.Type)
				require.NotNil(t, result.ResponseFormat.JSONSchema)
				assert.Equal(t, "my_schema", result.ResponseFormat.JSONSchema.Name)
				assert.Equal(t, "A test schema", result.ResponseFormat.JSONSchema.Description)
				require.NotNil(t, result.ResponseFormat.JSONSchema.Strict)
				assert.True(t, *result.ResponseFormat.JSONSchema.Strict)
				require.NotNil(t, result.ResponseFormat.JSONSchema.Schema)
				// Verify schema content round-tripped correctly
				var schemaMap map[string]interface{}
				err := json.Unmarshal(result.ResponseFormat.JSONSchema.Schema, &schemaMap)
				require.NoError(t, err)
				assert.Equal(t, "object", schemaMap["type"])
				props, ok := schemaMap["properties"].(map[string]interface{})
				require.True(t, ok)
				assert.Contains(t, props, "answer")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := makeReq(tt.responseFormat)
			result, err := ToHuggingFaceChatCompletionRequest(req)
			if tt.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			require.NotNil(t, result)
			tt.validate(t, result)
		})
	}
}
