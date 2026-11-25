import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  ChevronsLeftRightIcon,
  DownloadIcon,
  Maximize2Icon,
  Minimize2Icon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { clampZoom, ViewerToolButton } from '@/components/image-viewer'
import { useControllableState } from '@/hooks/use-controllable-state'
import { cn } from '@/lib/utils'

interface Props {
  before: string
  after: string
  beforeLabel?: string
  afterLabel?: string
  className?: string
  actions?: ReactNode
  zoom?: number | null
  onZoomChange?: (zoom: number | null) => void
}

export function CompareView({
  before,
  after,
  beforeLabel = 'Input',
  afterLabel = 'Output',
  className,
  actions,
  zoom: zoomProp,
  onZoomChange,
}: Props) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const imgBoxRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [zoom, setZoom] = useControllableState<number | null>({
    prop: zoomProp,
    defaultProp: null,
    onChange: onZoomChange,
    caller: 'CompareView',
  })
  const [pos, setPos] = useState(50)
  const [fullscreen, setFullscreen] = useState(false)
  const [prevSrc, setPrevSrc] = useState(after)

  if (prevSrc !== after) {
    setPrevSrc(after)
    if (zoomProp === undefined) setZoom(null)
  }

  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const update = () => setBox({ w: el.clientWidth, h: el.clientHeight })
    update()
    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const fitScale =
    natural && box.w > 0 && box.h > 0
      ? Math.min(box.w / natural.w, box.h / natural.h)
      : 1
  const scale = zoom ?? fitScale
  const percent = Math.round(scale * 100)

  useEffect(() => {
    const el = canvasRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()
      setZoom((z) =>
        clampZoom((z ?? fitScale) * (e.deltaY < 0 ? 1.15 : 1 / 1.15)),
      )
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [fitScale, setZoom])

  useEffect(() => {
    if (!fullscreen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setFullscreen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [fullscreen])

  const download = () => {
    const a = document.createElement('a')
    a.href = after
    a.download = `${afterLabel || 'output'}.png`
    a.click()
  }

  const updatePos = (clientX: number) => {
    const r = imgBoxRef.current?.getBoundingClientRect()
    if (!r || r.width === 0) return
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)))
  }

  const imgProps = {
    draggable: false,
    style: {
      imageRendering: scale >= 3 ? ('pixelated' as const) : ('auto' as const),
    },
  }

  return (
    <div
      className={cn(
        'flex flex-col overflow-hidden bg-neutral-950',
        fullscreen ? 'fixed inset-0 z-50' : 'rounded-xl',
        !fullscreen && className,
      )}
    >
      <div className="flex items-center gap-0.5 border-b border-white/10 bg-neutral-900 px-2 py-1.5">
        <ViewerToolButton
          label="Zoom out"
          onClick={() => setZoom(clampZoom(scale / 1.25))}
        >
          <ZoomOutIcon />
        </ViewerToolButton>
        <span className="w-14 text-center font-mono text-xs text-neutral-300 tabular-nums">
          {percent}%
        </span>
        <ViewerToolButton
          label="Zoom in"
          onClick={() => setZoom(clampZoom(scale * 1.25))}
        >
          <ZoomInIcon />
        </ViewerToolButton>
        <div className="mx-1 h-4 w-px bg-white/10" />
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={() => setZoom(null)}
          className={cn(
            'px-2 font-mono text-[11px] text-neutral-400 hover:bg-white/10 hover:text-white',
            zoom === null && 'bg-white/10 text-white',
          )}
        >
          Fit
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={() => setZoom(1)}
          className={cn(
            'px-2 font-mono text-[11px] text-neutral-400 hover:bg-white/10 hover:text-white',
            zoom === 1 && 'bg-white/10 text-white',
          )}
        >
          1:1
        </Button>
        <div className="ml-auto flex items-center gap-0.5">
          {actions}
          <ViewerToolButton label="Download image" onClick={download}>
            <DownloadIcon />
          </ViewerToolButton>
          <ViewerToolButton
            label={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            onClick={() => setFullscreen((f) => !f)}
          >
            {fullscreen ? <Minimize2Icon /> : <Maximize2Icon />}
          </ViewerToolButton>
        </div>
      </div>

      <div
        ref={canvasRef}
        className="checkerboard relative min-h-0 flex-1 overflow-auto"
      >
        <div className="flex min-h-full w-max min-w-full items-center justify-center">
          <div
            ref={imgBoxRef}
            role="slider"
            aria-label="Compare before and after image"
            aria-valuenow={Math.round(pos)}
            aria-valuemin={0}
            aria-valuemax={100}
            tabIndex={0}
            onPointerDown={(e) => {
              e.currentTarget.setPointerCapture(e.pointerId)
              updatePos(e.clientX)
            }}
            onPointerMove={(e) => {
              if (e.buttons === 1) updatePos(e.clientX)
            }}
            onKeyDown={(e) => {
              if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 5))
              if (e.key === 'ArrowRight') setPos((p) => Math.min(100, p + 5))
            }}
            className="relative cursor-ew-resize touch-none select-none focus-visible:outline-2 focus-visible:outline-ring"
            style={{
              width: natural ? natural.w * scale : undefined,
              height: natural ? natural.h * scale : undefined,
            }}
          >
            <img
              src={after}
              alt={afterLabel}
              onLoad={(e) =>
                setNatural({
                  w: e.currentTarget.naturalWidth,
                  h: e.currentTarget.naturalHeight,
                })
              }
              className="block h-full w-full object-contain"
              {...imgProps}
            />
            <div
              className="absolute inset-0"
              style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
            >
              <img
                src={before}
                alt={beforeLabel}
                className="h-full w-full object-contain"
                {...imgProps}
              />
            </div>

            <div
              className="absolute inset-y-0"
              style={{ left: `${pos}%` }}
            >
              <div className="absolute inset-y-0 -left-px w-0.5 bg-white shadow-[0_0_4px_rgba(0,0,0,0.6)]" />
              <div className="absolute top-1/2 grid size-7 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full bg-white text-neutral-900 shadow-md">
                <ChevronsLeftRightIcon className="size-4" />
              </div>
            </div>

            <span className="absolute top-2 left-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[11px] font-medium text-white">
              {beforeLabel}
            </span>
            <span className="absolute top-2 right-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[11px] font-medium text-white">
              {afterLabel}
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between gap-4 border-t border-white/10 bg-neutral-900 px-3 py-1 font-mono text-[11px] text-neutral-500">
        <span>{natural ? `${natural.w} x ${natural.h}px` : ''}</span>
        <span>
          {zoom === null ? `Fit ${percent}%` : `${percent}%`}
          {' · Ctrl + scroll to zoom'}
        </span>
      </div>
    </div>
  )
}
