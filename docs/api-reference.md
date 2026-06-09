# API Reference

Imports come from `quantvec` (isomorphic) and `quantvec/node` (Node filesystem helpers).

## `TurboQuantIndex`

A growable, positional flat quantized index.

```ts
new TurboQuantIndex(options: TurboQuantIndexOptions)
```

| Option      | Type                               | Default    | Notes                                                                                                                                                   |
| ----------- | ---------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dim`       | `number`                           | —          | positive multiple of 8                                                                                                                                  |
| `bits`      | `2 \| 3 \| 4`                      | `4`        | quantizer bit-width                                                                                                                                     |
| `metric`    | `'cosine' \| 'dot' \| 'euclidean'` | `'cosine'` | default ranking metric                                                                                                                                  |
| `seed`      | `number`                           | `0`        | rotation RNG seed (finite; truncated to an integer)                                                                                                     |
| `calibrate` | `boolean`                          | `false`    | opt-in TQ+ per-coordinate calibration (fit from the first ≥1000-vector add; data-dependent)                                                             |
| `wasm`      | `boolean`                          | `true`     | use the WASM scoring kernel when available (exact; auto-falls back to the scalar scan)                                                                  |
| `fastscan`  | `boolean`                          | `false`    | use the v128 FastScan kernel (4-bit only; approximate SIMD ranking + exact rescore; falls back to the exact kernel when `bits ≠ 4` or WASM unavailable) |

**Methods & getters**

| Member                                                     | Signature                                                                     | Description                                   |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------- | --------------------------------------------- |
| `add`                                                      | `(vectors: Float32Array \| number[][] \| Float32Array[]) => void`             | append a batch                                |
| `addOne`                                                   | `(vec: Float32Array \| number[]) => void`                                     | append one                                    |
| `search`                                                   | `(query: Float32Array, k: number, opts?: IndexSearchOptions) => SearchResult` | k nearest                                     |
| `swapRemove`                                               | `(i: number) => void`                                                         | O(1) delete; moves the last row into slot `i` |
| `clear`                                                    | `() => void`                                                                  | drop all vectors (keeps capacity)             |
| `toBytes`                                                  | `() => Uint8Array`                                                            | serialize                                     |
| `TurboQuantIndex.fromBytes`                                | `(bytes: Uint8Array) => TurboQuantIndex`                                      | deserialize (static)                          |
| `size` / `dim` / `bits` / `metric` / `seed` / `calibrated` | getters                                                                       | live count + config + whether TQ+ is active   |

`IndexSearchOptions`: `{ metric?: Distance; mask?: Uint8Array \| boolean[] }` (mask length = `size`,
positional). `SearchResult`: `{ indices: Int32Array; scores: Float32Array }`.

## `IdMapIndex<Id>`

Stable-id layer over `TurboQuantIndex`. `Id extends number | string | bigint`, default `number`.

```ts
new IdMapIndex<Id = number>(options: TurboQuantIndexOptions)
```

| Member                                                     | Signature                                                                               | Description                                 |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------- |
| `addWithIds`                                               | `(ids: readonly Id[], vectors: Float32Array \| number[][] \| Float32Array[]) => void`   | append with ids                             |
| `search`                                                   | `(query: Float32Array, k: number, opts?: IdMapSearchOptions<Id>) => IdSearchResult<Id>` | k nearest, by id                            |
| `has`                                                      | `(id: Id) => boolean`                                                                   | membership                                  |
| `remove`                                                   | `(id: Id) => void`                                                                      | O(1) delete by id                           |
| `ids`                                                      | `() => Id[]`                                                                            | snapshot of all ids (slot order)            |
| `clear`                                                    | `() => void`                                                                            | empty the index                             |
| `toBytes`                                                  | `() => Uint8Array`                                                                      | serialize                                   |
| `IdMapIndex.fromBytes`                                     | `<Id>(bytes: Uint8Array) => IdMapIndex<Id>`                                             | deserialize (static; assert `Id`)           |
| `size` / `dim` / `bits` / `metric` / `seed` / `calibrated` | getters                                                                                 | live count + config + whether TQ+ is active |

`IdMapSearchOptions<Id>`: `{ metric?: Distance; filter?: (id: Id) => boolean }`.
`IdSearchResult<Id>`: `{ ids: Id[]; scores: Float32Array }`.

## `quantvec/node`

| Function         | Signature                                                           |
| ---------------- | ------------------------------------------------------------------- |
| `saveIndex`      | `(index: { toBytes(): Uint8Array }, path: string) => Promise<void>` |
| `loadIndex`      | `(path: string) => Promise<TurboQuantIndex>`                        |
| `loadIdMapIndex` | `(path: string) => Promise<IdMapIndex<IdType>>`                     |
| `readIndexBytes` | `(path: string) => Promise<Uint8Array>`                             |

## Types & errors

- `Distance = 'cosine' | 'dot' | 'euclidean'`, `Bits = 2 | 3 | 4`, `IdType = number | string | bigint`.
- Errors (all exported, with a discriminated `.code`): `IndexError`, `IdMapError`, `DeserializeError`,
  `EncodeError`, `SearchError`. See the [Usage Guide](guide.md#errors) for the code tables.

## Constraints

- `dim` is a positive multiple of 8; `bits ∈ {2,3,4}`; `seed` is finite.
- Vectors must be finite and non-zero (a zero vector has no direction to quantize → `EncodeError`).
- `euclidean` scores are squared L2 distance (smaller = nearer); `cosine`/`dot` scores are similarities.
