package replicate

import (
	"fmt"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func newUpscaleRequest(extraParams map[string]interface{}) *schemas.BifrostImageGenerationRequest {
	return &schemas.BifrostImageGenerationRequest{
		Model: "prunaai/p-image-upscale",
		Params: &schemas.ImageGenerationParameters{
			ExtraParams: extraParams,
		},
	}
}

func TestResolveUpscaleOutputPixels_TargetMode(t *testing.T) {
	req := newUpscaleRequest(map[string]interface{}{
		"target":       float64(16),
		"upscale_mode": "target",
	})
	got := resolveUpscaleOutputPixels(req, nil)
	want := 16_000_000
	if got != want {
		t.Errorf("target=16 upscale_mode=target: want %d pixels, got %d", want, got)
	}
}

func TestResolveUpscaleOutputPixels_TargetMode_DefaultsWhenModeOmitted(t *testing.T) {
	// upscale_mode defaults to "target" per Replicate's docs, so a bare
	// "target" param (no explicit upscale_mode) must still resolve.
	req := newUpscaleRequest(map[string]interface{}{
		"target": float64(8),
	})
	got := resolveUpscaleOutputPixels(req, nil)
	want := 8_000_000
	if got != want {
		t.Errorf("target=8 (no upscale_mode): want %d pixels, got %d", want, got)
	}
}

func TestResolveUpscaleOutputPixels_FactorMode_IgnoresTargetFallsBackToMetrics(t *testing.T) {
	// A stray "target" value must not be used once upscale_mode explicitly
	// says "factor" — factor mode's output size depends on the input image,
	// not the target param.
	req := newUpscaleRequest(map[string]interface{}{
		"target":       float64(16), // should be ignored
		"upscale_mode": "factor",
		"factor":       float64(4),
	})
	resolutionTarget := "8-16MP"
	prediction := &ReplicatePredictionResponse{
		Metrics: &ReplicateMetrics{ResolutionTarget: &resolutionTarget},
	}
	got := resolveUpscaleOutputPixels(req, prediction)
	want := 16_000_000 // upper bound of the "8-16MP" band, not the ignored target=16
	if got != want {
		t.Errorf("factor mode: want %d pixels (from metrics band), got %d", want, got)
	}
}

func TestResolveUpscaleOutputPixels_NoSignal(t *testing.T) {
	req := newUpscaleRequest(nil)
	if got := resolveUpscaleOutputPixels(req, nil); got != 0 {
		t.Errorf("no target, no metrics: want 0, got %d", got)
	}
}

func TestResolveUpscaleOutputPixelsFromMetrics(t *testing.T) {
	cases := []struct {
		name string
		band *string
		want int
	}{
		{"range band", strPtr("8-16MP"), 16_000_000},
		{"single value band", strPtr("16MP"), 16_000_000},
		{"lowercase mp", strPtr("4-8mp"), 8_000_000},
		{"nil metrics field", nil, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			prediction := &ReplicatePredictionResponse{
				Metrics: &ReplicateMetrics{ResolutionTarget: tc.band},
			}
			got := resolveUpscaleOutputPixelsFromMetrics(prediction)
			if got != tc.want {
				t.Errorf("want %d, got %d", tc.want, got)
			}
		})
	}
	// nil prediction / nil metrics must not panic.
	if got := resolveUpscaleOutputPixelsFromMetrics(nil); got != 0 {
		t.Errorf("nil prediction: want 0, got %d", got)
	}
	if got := resolveUpscaleOutputPixelsFromMetrics(&ReplicatePredictionResponse{}); got != 0 {
		t.Errorf("nil metrics: want 0, got %d", got)
	}
}

func TestFormatSquarePixelSize_NeverUndercountsAtBoundary(t *testing.T) {
	// 8,000,000 has no integer square root (sqrt ≈ 2828.43); ceil-rounding
	// must keep width*height >= 8,000,000 so an exact-boundary target
	// (e.g. target: 8) never gets miscategorized into the tier below it.
	size := formatSquarePixelSize(8_000_000)
	w, h := parseWxH(t, size)
	if w*h < 8_000_000 {
		t.Errorf("formatSquarePixelSize(8_000_000) = %q, width*height = %d, want >= 8_000_000", size, w*h)
	}

	// A perfect square should round-trip exactly.
	size = formatSquarePixelSize(16_000_000)
	if size != "4000x4000" {
		t.Errorf("formatSquarePixelSize(16_000_000) = %q, want 4000x4000", size)
	}
}

func TestApplyUpscaleOutputResolution(t *testing.T) {
	req := newUpscaleRequest(map[string]interface{}{
		"target":       float64(16),
		"upscale_mode": "target",
	})
	resp := &schemas.BifrostImageGenerationResponse{}
	applyUpscaleOutputResolution(req, nil, resp)

	if resp.ImageGenerationResponseParameters == nil || resp.ImageGenerationResponseParameters.Size == "" {
		t.Fatal("expected Size to be backfilled")
	}
	w, h := parseWxH(t, resp.ImageGenerationResponseParameters.Size)
	if w*h < 16_000_000 {
		t.Errorf("backfilled size %q has %d total pixels, want >= 16_000_000", resp.ImageGenerationResponseParameters.Size, w*h)
	}
}

func TestApplyUpscaleOutputResolution_DoesNotOverwriteExistingSize(t *testing.T) {
	req := newUpscaleRequest(map[string]interface{}{"target": float64(16)})
	resp := &schemas.BifrostImageGenerationResponse{
		ImageGenerationResponseParameters: &schemas.ImageGenerationResponseParameters{Size: "1234x1234"},
	}
	applyUpscaleOutputResolution(req, nil, resp)

	if resp.ImageGenerationResponseParameters.Size != "1234x1234" {
		t.Errorf("existing Size was overwritten: got %q", resp.ImageGenerationResponseParameters.Size)
	}
}

func TestApplyUpscaleOutputResolution_NoSignalLeavesSizeEmpty(t *testing.T) {
	req := newUpscaleRequest(nil)
	resp := &schemas.BifrostImageGenerationResponse{}
	applyUpscaleOutputResolution(req, nil, resp)

	if resp.ImageGenerationResponseParameters != nil && resp.ImageGenerationResponseParameters.Size != "" {
		t.Errorf("expected no Size to be set, got %q", resp.ImageGenerationResponseParameters.Size)
	}
}

func strPtr(s string) *string { return &s }

func parseWxH(t *testing.T, size string) (int, int) {
	t.Helper()
	var w, h int
	n, err := fmt.Sscanf(size, "%dx%d", &w, &h)
	if err != nil || n != 2 {
		t.Fatalf("could not parse size %q: %v", size, err)
	}
	return w, h
}
