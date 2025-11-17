package graph

import (
	"fmt"
	"reflect"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/graph"
	"google.golang.org/protobuf/proto"
)

// config holds graph marshaller-specific configuration
type config struct {
	nodePath        string
	edgePath        string
	dataPath        string
	mirror          bool
	readOnly        bool
	onEqual         func(a, b graph.Node[any, any]) bool
	onCompare       func(a, b graph.Node[any, any]) int
	onCost          func(from, to graph.Node[any, any]) float32
	registeredTypes map[string]proto.Message
	decisionNodeOps map[string]decisionNodeOpInfo
	decisionEdgeOps map[string]decisionEdgeOpInfo
	expressionOps   map[string]expressionOpInfo
}

// withNodePath sets the path for the node file
type withNodePath struct {
	path string
}

func (opt withNodePath) Apply(opts *types.Options) {
	// Store in metadata for now, will extract in marshaller
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	opts.Metadata["graph.nodePath"] = opt.path
}

// WithPath sets the base path for graph files.
// This sets the node file path. Edge and data paths default to
// <base>.edges.graph and <base>.data.graph
func WithPath(path string) types.Option {
	return withNodePath{path: path}
}

// withEdgePath sets the path for the edge file
type withEdgePath struct {
	path string
}

func (opt withEdgePath) Apply(opts *types.Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	opts.Metadata["graph.edgePath"] = opt.path
}

// WithEdgesPath sets the path for the edge file
func WithEdgesPath(path string) types.Option {
	return withEdgePath{path: path}
}

// withDataPath sets the path for the data file
type withDataPath struct {
	path string
}

func (opt withDataPath) Apply(opts *types.Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	opts.Metadata["graph.dataPath"] = opt.path
}

// WithLabelsPath sets the path for the data file (node/edge labels)
func WithLabelsPath(path string) types.Option {
	return withDataPath{path: path}
}

// withMirror sets whether to link unmarshalled graph to storage
type withMirror struct {
	mirror bool
}

func (opt withMirror) Apply(opts *types.Options) {
	if opts.Metadata == nil {
		opts.Metadata = make(map[string]string)
	}
	if opt.mirror {
		opts.Metadata["graph.mirror"] = "true"
	} else {
		opts.Metadata["graph.mirror"] = "false"
	}
}

// WithMirror sets whether to link unmarshalled graph to storage.
// Default is true (linked).
func WithMirror(mirror bool) types.Option {
	return withMirror{mirror: mirror}
}

// withOnEqual sets the callback for node equality comparison
type withOnEqual struct {
	fn func(a, b graph.Node[any, any]) bool
}

func (opt withOnEqual) Apply(opts *types.Options) {
	// Callback will be extracted in applyOptions by checking option types
	// No need to store in metadata
}

// WithEqual sets the callback function for node equality comparison.
// The callback receives two nodes and returns true if they are equal.
// This is used by StoredNode.Equal() method.
func WithEqual(fn func(a, b graph.Node[any, any]) bool) types.Option {
	return withOnEqual{fn: fn}
}

// withOnCompare sets the callback for node comparison
type withOnCompare struct {
	fn func(a, b graph.Node[any, any]) int
}

func (opt withOnCompare) Apply(opts *types.Options) {
	// Callback will be extracted in applyOptions by checking option types
}

// WithCompare sets the callback function for node comparison.
// The callback receives two nodes and returns:
//   - -1 if a < b
//   - 0 if a == b
//   - +1 if a > b
//
// This is used by StoredNode.Compare() method.
func WithCompare(fn func(a, b graph.Node[any, any]) int) types.Option {
	return withOnCompare{fn: fn}
}

// withOnCost sets the callback for edge cost calculation
type withOnCost struct {
	fn func(from, to graph.Node[any, any]) float32
}

func (opt withOnCost) Apply(opts *types.Options) {
	// Callback will be extracted in applyOptions by checking option types
}

// WithCost sets the callback function for edge cost calculation.
// The callback receives from and to nodes and returns the cost.
// This is used by StoredNode.Cost() and StoredEdge.Cost() methods.
func WithCost(fn func(from, to graph.Node[any, any]) float32) types.Option {
	return withOnCost{fn: fn}
}

// withDecisionOp registers decision node operations
type withDecisionOp struct {
	ops []struct {
		name string
		info decisionNodeOpInfo
	}
}

func (opt withDecisionOp) Apply(opts *types.Options) {}

// WithDecisionOp registers one or more decision node operations by reflection-derived name.
// Each function must have signature `func(Input) (Output, bool)`.
func WithDecisionOp(fns ...any) types.Option {
	opt := withDecisionOp{ops: make([]struct {
		name string
		info decisionNodeOpInfo
	}, 0, len(fns))}
	for _, fn := range fns {
		info, name, err := newDecisionNodeOpInfo(fn)
		if err != nil {
			panic(fmt.Errorf("graph.WithDecisionOp: %w", err))
		}
		opt.ops = append(opt.ops, struct {
			name string
			info decisionNodeOpInfo
		}{name: name, info: info})
	}
	return opt
}

// withDecisionEdge registers decision edge criteria functions
type withDecisionEdge struct {
	ops []struct {
		name string
		info decisionEdgeOpInfo
	}
}

func (opt withDecisionEdge) Apply(opts *types.Options) {}

// WithDecisionEdge registers one or more decision edge operations (criteria evaluators).
// Each function must have signature `func(Input) bool`.
func WithDecisionEdge(fns ...any) types.Option {
	opt := withDecisionEdge{ops: make([]struct {
		name string
		info decisionEdgeOpInfo
	}, 0, len(fns))}
	for _, fn := range fns {
		info, name, err := newDecisionEdgeOpInfo(fn)
		if err != nil {
			panic(fmt.Errorf("graph.WithDecisionEdge: %w", err))
		}
		opt.ops = append(opt.ops, struct {
			name string
			info decisionEdgeOpInfo
		}{name: name, info: info})
	}
	return opt
}

// withExpressionOp registers expression node computation functions
type withExpressionOp struct {
	ops []struct {
		name string
		info expressionOpInfo
	}
}

func (opt withExpressionOp) Apply(opts *types.Options) {}

// WithExpressionOp registers one or more expression node operations.
// Each function must have signature `func(Input, map[int64]Output) (Output, bool)`.
func WithExpressionOp(fns ...any) types.Option {
	opt := withExpressionOp{ops: make([]struct {
		name string
		info expressionOpInfo
	}, 0, len(fns))}
	for _, fn := range fns {
		info, name, err := newExpressionOpInfo(fn)
		if err != nil {
			panic(fmt.Errorf("graph.WithExpressionOp: %w", err))
		}
		opt.ops = append(opt.ops, struct {
			name string
			info expressionOpInfo
		}{name: name, info: info})
	}
	return opt
}

// withRegisteredType stores a proto.Message prototype keyed by its name.
type withRegisteredType struct {
	prototype proto.Message
}

func (opt withRegisteredType) Apply(opts *types.Options) {
	// Prototype handled in applyOptions
}

// WithType registers a proto.Message type that may appear in node or edge data.
// Provide this option multiple times to register multiple message types.
func WithType(example proto.Message) types.Option {
	return withRegisteredType{prototype: example}
}

// applyOptions extracts graph-specific options from types.Options
func applyOptions(baseOpts types.Options, baseCfg config, opts []types.Option) (types.Options, config) {
	// Start with base options
	resultOpts := baseOpts
	if resultOpts.Metadata == nil {
		resultOpts.Metadata = make(map[string]string)
	}

	// Extract callbacks/types from options before applying them
	// (since callbacks can't be stored in metadata)
	resultCfg := baseCfg
	if baseCfg.registeredTypes != nil {
		resultCfg.registeredTypes = make(map[string]proto.Message, len(baseCfg.registeredTypes))
		for name, prototype := range baseCfg.registeredTypes {
			if prototype != nil {
				resultCfg.registeredTypes[name] = proto.Clone(prototype)
			}
		}
	} else {
		resultCfg.registeredTypes = make(map[string]proto.Message)
	}
	if baseCfg.decisionNodeOps != nil {
		resultCfg.decisionNodeOps = make(map[string]decisionNodeOpInfo, len(baseCfg.decisionNodeOps))
		for name, info := range baseCfg.decisionNodeOps {
			resultCfg.decisionNodeOps[name] = info
		}
	} else {
		resultCfg.decisionNodeOps = make(map[string]decisionNodeOpInfo)
	}
	if baseCfg.decisionEdgeOps != nil {
		resultCfg.decisionEdgeOps = make(map[string]decisionEdgeOpInfo, len(baseCfg.decisionEdgeOps))
		for name, info := range baseCfg.decisionEdgeOps {
			resultCfg.decisionEdgeOps[name] = info
		}
	} else {
		resultCfg.decisionEdgeOps = make(map[string]decisionEdgeOpInfo)
	}
	if baseCfg.expressionOps != nil {
		resultCfg.expressionOps = make(map[string]expressionOpInfo, len(baseCfg.expressionOps))
		for name, info := range baseCfg.expressionOps {
			resultCfg.expressionOps[name] = info
		}
	} else {
		resultCfg.expressionOps = make(map[string]expressionOpInfo)
	}

	for _, opt := range opts {
		switch v := opt.(type) {
		case withOnEqual:
			resultCfg.onEqual = v.fn
		case withOnCompare:
			resultCfg.onCompare = v.fn
		case withOnCost:
			resultCfg.onCost = v.fn
		case withRegisteredType:
			if v.prototype == nil {
				break
			}
			if name := protoTypeName(v.prototype); name != "" {
				resultCfg.registeredTypes[name] = proto.Clone(v.prototype)
			}
		case withDecisionOp:
			for _, op := range v.ops {
				if op.name != "" {
					resultCfg.decisionNodeOps[op.name] = op.info
				}
			}
		case withDecisionEdge:
			for _, op := range v.ops {
				if op.name != "" {
					resultCfg.decisionEdgeOps[op.name] = op.info
				}
			}
		case withExpressionOp:
			for _, op := range v.ops {
				if op.name != "" {
					resultCfg.expressionOps[op.name] = op.info
				}
			}
		}
		opt.Apply(&resultOpts)
	}

	// Extract config from metadata
	if path, ok := resultOpts.Metadata["graph.nodePath"]; ok {
		resultCfg.nodePath = path
	}
	if path, ok := resultOpts.Metadata["graph.edgePath"]; ok {
		resultCfg.edgePath = path
	}
	if path, ok := resultOpts.Metadata["graph.dataPath"]; ok {
		resultCfg.dataPath = path
	}
	if mirror, ok := resultOpts.Metadata["graph.mirror"]; ok {
		resultCfg.mirror = (mirror == "true")
	} else {
		resultCfg.mirror = true // Default to true
	}

	// Set default paths if not specified
	if resultCfg.nodePath == "" {
		resultCfg.nodePath = "graph.nodes.graph"
	}
	if resultCfg.edgePath == "" {
		// Default to <base>.edges.graph
		if resultCfg.nodePath != "" && len(resultCfg.nodePath) > len(".nodes.graph") {
			resultCfg.edgePath = resultCfg.nodePath[:len(resultCfg.nodePath)-len(".nodes.graph")] + ".edges.graph"
		} else {
			resultCfg.edgePath = "graph.edges.graph"
		}
	}
	if resultCfg.dataPath == "" {
		// Default to <base>.data.graph
		if resultCfg.nodePath != "" && len(resultCfg.nodePath) > len(".nodes.graph") {
			resultCfg.dataPath = resultCfg.nodePath[:len(resultCfg.nodePath)-len(".nodes.graph")] + ".data.graph"
		} else {
			resultCfg.dataPath = "graph.data.graph"
		}
	}

	return resultOpts, resultCfg
}

func protoTypeName(msg proto.Message) string {
	if msg == nil {
		return ""
	}
	if name := string(proto.MessageName(msg)); name != "" {
		return name
	}
	t := reflect.TypeOf(msg)
	if t == nil {
		return ""
	}
	return t.String()
}
