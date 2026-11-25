import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  DownloadIcon,
  Maximize2Icon,
  Minimize2Icon,
  ZoomInIcon,
  ZoomOutIcon,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useControllableState } from '@/hooks/use-controllable-state'
import { cn } from '@/lib/utils'

const MIN_ZOOM = 0.05
const MAX_ZOOM = 16
export const clampZoom = (v: number) =>
  Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, v))

export function ViewerToolButton({
  label,
  active,
  onClick,
  children,
}: {
  label: string
  active?: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-xs"
      title={label}
      aria-label={label}
      onClick={onClick}
      className={cn(
        'text-neutral-400 hover:bg-white/10 hover:text-white',
        active && 'bg-white/10 text-white',
      )}
    >
      {children}
    </Button>
  )
}

interface Props {
  src: string
  alt: string
  className?: string
  actions?: ReactNode
  zoom?: number | null
  onZoomChange?: (zoom: number | null) => void
}

export function ImageViewer({ src, alt, className, actions, zoom: zoomProp, onZoomChange }: Props) {
  const canvasRef = useRef<HTMLDivElement>(null)
  const [box, setBox] = useState({ w: 0, h: 0 })
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [zoom, setZoom] = useControllableState<number | null>({
    prop: zoomProp,
    defaultProp: null,
    onChange: onZoomChange,
    caller: 'ImageViewer',
  })
  const [fullscreen, setFullscreen] = useState(false)
  const [prevSrc, setPrevSrc] = useState(src)

  if (prevSrc !== src) {
    setPrevSrc(src)
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
    a.href = src
    a.download = `${alt || 'image'}.png`
    a.click()
  }

  const imgProps = {
    src,
    alt,
    draggable: false,
    onLoad: (e: React.SyntheticEvent<HTMLImageElement>) =>
      setNatural({
        w: e.currentTarget.naturalWidth,
        h: e.currentTarget.naturalHeight,
      }),
    style: { imageRendering: scale >= 3 ? ('pixelated' as const) : ('auto' as const) },
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
        {zoom === null ? (
          <img
            {...imgProps}
            className="absolute top-1/2 left-1/2 max-h-full max-w-full -translate-x-1/2 -translate-y-1/2"
          />
        ) : (
          <div className="flex min-h-full w-max min-w-full items-center justify-center">
            <img
              {...imgProps}
              className="max-w-none"
              style={{
                ...imgProps.style,
                width: natural ? natural.w * scale : undefined,
              }}
            />
          </div>
        )}
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
