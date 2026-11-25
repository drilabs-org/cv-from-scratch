import { useEffect, useState } from 'react'
import { ProportionsIcon, TimerIcon } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { CameraCapture } from '@/components/camera-capture'
import { CompareView } from '@/components/compare-view'
import { DropZone } from '@/components/drop-zone'
import { ImageViewer } from '@/components/image-viewer'
import { InputActions } from '@/components/input-actions'
import { cn } from '@/lib/utils'
import type { TaskResult } from '@/types'

type ProbRow = { label: unknown; prob: number }

function isProbRow(v: unknown): v is ProbRow {
  if (typeof v !== 'object' || v === null || !('label' in v)) return false
  const prob = (v as Record<string, unknown>).prob
  return typeof prob === 'number' && prob >= 0 && prob <= 1
}

function ConfidenceBars({ rows }: { rows: ProbRow[] }) {
  return (
    <div className="flex w-64 flex-col gap-1.5">
      {rows.map((row, i) => (
        <div key={i} className="flex items-center gap-2 text-sm">
          <span className="w-8 shrink-0 truncate font-mono font-semibold">
            {String(row.label)}
          </span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary"
              style={{ width: `${row.prob * 100}%` }}
            />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-xs text-muted-foreground">
            {(row.prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  )
}

function DataValue({ value }: { value: unknown }) {
  if (typeof value === 'number') {
    return (
      <Badge variant="secondary" className="text-base">
        {value}
      </Badge>
    )
  }
  if (Array.isArray(value) && value.length > 0 && value.every(isProbRow)) {
    return <ConfidenceBars rows={value} />
  }
  if (
    Array.isArray(value) &&
    value.every((v) => typeof v === 'object' && v !== null)
  ) {
    return (
      <div className="flex flex-col gap-1">
        {value.map((row, i) => (
          <div key={i} className="flex gap-3 text-sm">
            {Object.entries(row).map(([k, v]) => (
              <span key={k}>
                {k}: <strong>{String(v)}</strong>
              </span>
            ))}
          </div>
        ))}
      </div>
    )
  }
  return <span className="font-mono text-sm">{JSON.stringify(value)}</span>
}

type ViewTab = 'input' | 'compare' | 'output'
type FrameRatio = 'fit' | '1:1' | '4:3' | '16:9'

const FRAME_RATIOS: { key: FrameRatio; label: string }[] = [
  { key: 'fit', label: 'Fit' },
  { key: '1:1', label: '1:1' },
  { key: '4:3', label: '4:3' },
  { key: '16:9', label: '16:9' },
]

const FIXED_RATIO_CLS: Record<Exclude<FrameRatio, 'fit'>, string> = {
  '1:1': 'aspect-square',
  '4:3': 'aspect-[4/3]',
  '16:9': 'aspect-video',
}

export function ResultView({
  result,
  loading,
  inputUrl,
  fileName,
  onFile,
  onClear,
  elapsedMs,
  preferredTab = 'auto',
}: {
  result: TaskResult | null
  loading: boolean
  inputUrl: string | null
  fileName?: string | null
  onFile: (f: File) => void
  onClear?: () => void
  elapsedMs?: number | null
  preferredTab?: 'auto' | ViewTab
}) {
  const [tab, setTab] = useState<ViewTab>('input')
  const [cameraOpen, setCameraOpen] = useState(false)
  const [prevResult, setPrevResult] = useState(result)
  const [frameRatio, setFrameRatio] = useState<FrameRatio>('16:9')
  const [viewerZoom, setViewerZoom] = useState<number | null>(null)
  const [inputAspect, setInputAspect] = useState<number | null>(null)
  const [outputAspect, setOutputAspect] = useState<number | null>(null)

  const imageUri = result?.image
    ? `data:image/png;base64,${result.image}`
    : null
  const canCompare = Boolean(imageUri && inputUrl)
  const dataEntries = result ? Object.entries(result.data) : []

  if (prevResult !== result) {
    setPrevResult(result)
    setTab((t) => {
      if (!result) return 'input'
      const stillValid =
        t === 'input' || t === 'output' || (t === 'compare' && canCompare)
      if (prevResult !== null && stillValid) return t
      if (preferredTab !== 'auto' && (preferredTab !== 'compare' || canCompare)) {
        return preferredTab
      }
      return 'output'
    })
  }

  const [prevInputUrl, setPrevInputUrl] = useState(inputUrl)
  if (prevInputUrl !== inputUrl) {
    setPrevInputUrl(inputUrl)
    setInputAspect(null)
    setViewerZoom(null)
  }
  const [prevImageUri, setPrevImageUri] = useState(imageUri)
  if (prevImageUri !== imageUri) {
    setPrevImageUri(imageUri)
    setOutputAspect(null)
  }

  useEffect(() => {
    if (!inputUrl) return
    const img = new Image()
    img.onload = () => setInputAspect(img.naturalWidth / img.naturalHeight)
    img.src = inputUrl
  }, [inputUrl])

  useEffect(() => {
    if (!imageUri) return
    const img = new Image()
    img.onload = () => setOutputAspect(img.naturalWidth / img.naturalHeight)
    img.src = imageUri
  }, [imageUri])

  const tabs: { key: ViewTab; label: string; enabled: boolean }[] = [
    { key: 'input', label: 'Input', enabled: true },
    { key: 'output', label: 'Output', enabled: Boolean(result) },
    { key: 'compare', label: 'Compare', enabled: canCompare },
  ]

  const aspect =
    tab === 'input' ? inputAspect : (outputAspect ?? inputAspect)
  const frameCls =
    frameRatio === 'fit'
      ? 'relative w-full'
      : cn('relative h-auto w-full', FIXED_RATIO_CLS[frameRatio])
  const frameStyle =
    frameRatio === 'fit'
      ? { aspectRatio: String(aspect ?? 4 / 3), maxHeight: '75vh' }
      : undefined

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <div
          role="tablist"
          className="inline-flex rounded-lg bg-muted p-0.5"
        >
          {tabs.map((t) => (
            <button
              key={t.key}
              type="button"
              role="tab"
              aria-selected={tab === t.key}
              disabled={!t.enabled}
              onClick={() => setTab(t.key)}
              className={cn(
                'rounded-md px-3 py-1 text-xs font-medium transition-colors',
                tab === t.key
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
                !t.enabled &&
                  'cursor-not-allowed opacity-40 hover:text-muted-foreground',
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="ml-auto inline-flex items-center rounded-lg bg-muted p-0.5">
          <ProportionsIcon className="mx-1.5 size-3.5 shrink-0 text-muted-foreground" />
          {FRAME_RATIOS.map((r) => (
            <button
              key={r.key}
              type="button"
              aria-pressed={frameRatio === r.key}
              onClick={() => setFrameRatio(r.key)}
              className={cn(
                'rounded-md px-2 py-1 font-mono text-[11px] transition-colors',
                frameRatio === r.key
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {r.label}
            </button>
          ))}
        </div>
        {elapsedMs != null && (
          <Badge variant="secondary" className="font-mono">
            <TimerIcon data-icon="inline-start" />
            {elapsedMs < 1000
              ? `${Math.round(elapsedMs)} ms`
              : `${(elapsedMs / 1000).toFixed(2)} s`}
          </Badge>
        )}
      </div>

      <div className={frameCls} style={frameStyle}>
        <div className="absolute inset-0">
        {cameraOpen ? (
          <CameraCapture
            className="h-full w-full"
            onCapture={(f) => {
              onFile(f)
              setCameraOpen(false)
            }}
            onClose={() => setCameraOpen(false)}
          />
        ) : tab === 'compare' && canCompare && inputUrl && imageUri ? (
          <CompareView
            before={inputUrl}
            after={imageUri}
            beforeLabel="Input"
            afterLabel="Output"
            className="h-full w-full"
            actions={
              <InputActions
                onFile={onFile}
                onClear={onClear}
                onOpenCamera={() => setCameraOpen(true)}
              />
            }
            zoom={viewerZoom}
            onZoomChange={setViewerZoom}
          />
        ) : tab === 'output' && result ? (
          imageUri ? (
            <ImageViewer
              src={imageUri}
              alt="result"
              className="h-full w-full"
              actions={
                <InputActions
                  onFile={onFile}
                  onClear={onClear}
                  onOpenCamera={() => setCameraOpen(true)}
                />
              }
              zoom={viewerZoom}
              onZoomChange={setViewerZoom}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center rounded-xl border-2 border-dashed border-border">
              <p className="text-sm text-muted-foreground">
                The task returned no image output.
              </p>
            </div>
          )
        ) : (
          <DropZone
            previewUrl={inputUrl}
            fileName={fileName}
            onFile={onFile}
            onClear={onClear}
            onOpenCamera={() => setCameraOpen(true)}
            className="h-full w-full"
            zoom={viewerZoom}
            onZoomChange={setViewerZoom}
          />
        )}
        {loading && (
          <Skeleton className="absolute inset-0 z-10 h-full w-full rounded-xl" />
        )}
        </div>
      </div>

      {imageUri && dataEntries.length > 0 && <Separator />}
      {dataEntries.length > 0 && (
        <div className="flex flex-col gap-2">
          {dataEntries.map(([k, v]) => (
            <div key={k} className="flex items-start justify-between gap-4">
              <span className="text-sm text-muted-foreground">{k}</span>
              <DataValue value={v} />
            </div>
          ))}
        </div>
      )}
      {result && !imageUri && dataEntries.length === 0 && (
        <p className="text-sm text-muted-foreground">
          The task returned no output.
        </p>
      )}
    </div>
  )
}
