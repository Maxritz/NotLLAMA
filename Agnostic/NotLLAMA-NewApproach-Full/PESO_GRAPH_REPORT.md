# Graph Report - .  (2026-06-28)

## Corpus Check
- 26 files · ~146,824 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 331 nodes · 512 edges · 14 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Prompt Enhancement|Prompt Enhancement]]
- [[_COMMUNITY_Schema Definitions|Schema Definitions]]
- [[_COMMUNITY_Package Configuration|Package Configuration]]
- [[_COMMUNITY_Intent Classification|Intent Classification]]
- [[_COMMUNITY_CLI Interface|CLI Interface]]
- [[_COMMUNITY_Context Management|Context Management]]
- [[_COMMUNITY_Schema Defaults|Schema Defaults]]
- [[_COMMUNITY_Schema Properties|Schema Properties]]
- [[_COMMUNITY_Configuration System|Configuration System]]
- [[_COMMUNITY_Context Gathering|Context Gathering]]
- [[_COMMUNITY_Research Rules|Research Rules]]
- [[_COMMUNITY_Schema Flags|Schema Flags]]
- [[_COMMUNITY_TypeScript Config|TypeScript Config]]

## God Nodes (most connected - your core abstractions)
1. `runPipeline()` - 17 edges
2. `main()` - 15 edges
3. `gatherContext()` - 12 edges
4. `scorePrompt()` - 12 edges
5. `classify()` - 11 edges
6. `resolveConfig()` - 11 edges
7. `compilerOptions` - 10 edges
8. `checkAllRules()` - 9 edges
9. `callSmallModel()` - 8 edges
10. `PesoPlugin()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `EnhancementOptions` --references--> `Domain`  [EXTRACTED]
  src/enhancer.ts → src/classifier.ts
- `main()` --calls--> `classify()`  [EXTRACTED]
  src/cli.ts → src/classifier.ts
- `main()` --calls--> `generateClarifyingQuestions()`  [EXTRACTED]
  src/cli.ts → src/classifier.ts
- `main()` --calls--> `formatContextBlock()`  [EXTRACTED]
  src/cli.ts → src/context-gatherer.ts
- `main()` --calls--> `gatherContext()`  [EXTRACTED]
  src/cli.ts → src/context-gatherer.ts

## Import Cycles
- None detected.

## Communities (14 total, 0 thin omitted)

### Community 0 - "Prompt Enhancement"
Cohesion: 0.09
Nodes (30): AgentVector, autonomyFromPermissions(), buildAgentVector(), buildSystemPrompt(), ContextConfig, __dirname, _extraTechniques, getIntensity() (+22 more)

### Community 1 - "Schema Definitions"
Cohesion: 0.06
Nodes (30): additionalProperties, description, type, description, type, additionalProperties, required, type (+22 more)

### Community 2 - "Package Configuration"
Cohesion: 0.07
Nodes (28): dependencies, @opencode-ai/plugin, description, devDependencies, eslint, @eslint/js, globals, prettier (+20 more)

### Community 3 - "Intent Classification"
Cohesion: 0.11
Nodes (27): AMBIGUITY_INDICATORS, ArgumentSlots, ClassificationResult, classify(), Complexity, decideRouting(), detectAmbiguity(), detectComplexity() (+19 more)

### Community 4 - "CLI Interface"
Cohesion: 0.14
Nodes (27): args, CliArgs, debug(), header(), kvLine(), log(), main(), parseArgs() (+19 more)

### Community 5 - "Context Management"
Cohesion: 0.07
Nodes (27): context, injectGitBranch, injectGitChangedFiles, injectGitLastCommit, injectMcpTools, injectProjectInstructions, maxChangedFiles, $schema (+19 more)

### Community 6 - "Schema Defaults"
Cohesion: 0.08
Nodes (27): type, default, description, items, type, items, type, default (+19 more)

### Community 7 - "Schema Properties"
Cohesion: 0.08
Nodes (26): properties, default, description, type, default, description, type, default (+18 more)

### Community 8 - "Configuration System"
Cohesion: 0.12
Nodes (22): deepMerge(), DEFAULTS, discoverGlobalPacks(), loadJsonFile(), loadPesoConfig(), PACKS_DIR, PesoConfig, resolveEnvTemplates() (+14 more)

### Community 9 - "Context Gathering"
Cohesion: 0.16
Nodes (17): cache, CacheEntry, ContextCache, ContextConfig, detectAvailableTools(), detectStaleInfo(), formatContextBlock(), gatherContext() (+9 more)

### Community 10 - "Research Rules"
Cohesion: 0.20
Nodes (16): checkAllRules(), checkDuplicateRules(), checkInstructionRatio(), checkNestingDepth(), checkPositionSensitivity(), checkPriorityClarity(), checkStepByStep(), RuleCheckResult (+8 more)

### Community 11 - "Schema Flags"
Cohesion: 0.14
Nodes (14): default, description, type, default, description, oneOf, type, disabled (+6 more)

### Community 12 - "TypeScript Config"
Cohesion: 0.15
Nodes (12): compilerOptions, esModuleInterop, module, moduleResolution, outDir, rootDir, skipLibCheck, strict (+4 more)

## Knowledge Gaps
- **164 isolated node(s):** `name`, `version`, `description`, `type`, `main` (+159 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `properties` connect `Schema Definitions` to `Schema Flags`, `Schema Defaults`?**
  _High betweenness centrality (0.066) - this node is a cross-community bridge._
- **Why does `properties` connect `Schema Properties` to `Schema Definitions`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `context` connect `Schema Definitions` to `Schema Properties`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **What connects `name`, `version`, `description` to the rest of the system?**
  _164 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Prompt Enhancement` be split into smaller, more focused modules?**
  _Cohesion score 0.09206349206349207 - nodes in this community are weakly interconnected._
- **Should `Schema Definitions` be split into smaller, more focused modules?**
  _Cohesion score 0.06451612903225806 - nodes in this community are weakly interconnected._
- **Should `Package Configuration` be split into smaller, more focused modules?**
  _Cohesion score 0.06896551724137931 - nodes in this community are weakly interconnected._