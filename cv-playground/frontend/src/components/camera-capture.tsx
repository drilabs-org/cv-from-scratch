import { useEffect, useRef, useState } from 'react'
import {
  CameraIcon,
  CameraOffIcon,
  SwitchCameraIcon,
  XIcon,
  ZoomInIcon,
} from 'lucide-react'
import { ViewerToolButton } from '@/components/image-viewer'
import { Slider } from '@/components/ui/slider'
import { Spinner } from '@/components/ui/spinner'
import { cn } from '@/lib/utils'

type Facing = 'user' | 'environment'

const ZOOM_MIN = 1
const ZOOM_MAX = 4

interface Props {
  onCapture: (f: File) => void
  onClose: () => void
  className?: string
}

export function CameraCapture({ onCapture, onClose, className }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [facing, setFacing] = useState<Facing>('environment')
  const [zoom, setZoom] = useState(1)
  const supported = Boolean(navigator.mediaDevices?.getUserMedia)

  useEffect(() => {
    if (!supported) return
    let cancelled = false
    let stream: MediaStream | null = null
    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: { ideal: facing } },
        audio: false,
      })
      .then((s) => {
        if (cancelled) {
          s.getTracks().forEach((t) => t.stop())
          return
        }
        stream = s
        const v = videoRef.current
        if (v) v.srcObject = s
      })
      .catch((e) => {
        if (cancelled) return
        setError(
          e instanceof DOMException && e.name === 'NotAllowedError'
            ? 'Camera permission was denied. Allow access in the browser and try again.'
            : e instanceof Error
              ? e.message
              : 'Could not open the camera.',
        )
      })
    return () => {
      cancelled = true
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [facing, supported])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const capture = () => {
    const video = videoRef.current
    if (!video || video.videoWidth === 0) return
    const sw = video.videoWidth / zoom
    const sh = video.videoHeight / zoom
    const sx = (video.videoWidth - sw) / 2
    const sy = (video.videoHeight - sh) / 2
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(sw)
    canvas.height = Math.round(sh)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    if (facing === 'user') {
      ctx.translate(canvas.width, 0)
      ctx.scale(-1, 1)
    }
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (!blob) return
      onCapture(
        new File([blob], `capture-${Date.now()}.png`, { type: 'image/png' }),
      )
    }, 'image/png')
  }

  const err =
    error ??
    (supported
      ? null
      : 'Camera is not available. It requires HTTPS or localhost and a browser with camera support.')

  return (
    <div
      className={cn(
        'flex flex-col overflow-hidden rounded-xl bg-neutral-950',
        className,
      )}
    >
      <div className="flex items-center gap-1 border-b border-white/10 bg-neutral-900 px-2 py-1.5">
        <CameraIcon className="mx-1 size-3.5 text-neutral-400" />
        <span className="font-mono text-[11px] text-neutral-300">Camera</span>
        <div className="ml-auto flex items-center gap-0.5">
          <ViewerToolButton
            label="Switch camera"
            onClick={() => {
              setReady(false)
              setError(null)
              setFacing((f) => (f === 'user' ? 'environment' : 'user'))
            }}
          >
            <SwitchCameraIcon />
          </ViewerToolButton>
          <ViewerToolButton label="Close camera" onClick={onClose}>
            <XIcon />
          </ViewerToolButton>
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-hidden">
        {err ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 p-4 text-center">
            <CameraOffIcon className="size-6 text-neutral-500" />
            <p className="max-w-xs text-xs text-neutral-400">{err}</p>
          </div>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              onLoadedData={() => setReady(true)}
              style={{ transform: zoom > 1 ? `scale(${zoom})` : undefined }}
              className={cn(
                'absolute inset-0 size-full object-contain',
                facing === 'user' && '-scale-x-100',
              )}
            />
            {!ready && (
              <div className="absolute inset-0 grid place-items-center">
                <Spinner className="text-neutral-400" />
              </div>
            )}
          </>
        )}
      </div>

      <div className="flex items-center gap-3 border-t border-white/10 bg-neutral-900 px-3 py-2">
        <div className="flex min-w-0 flex-1 items-center gap-2">
          <ZoomInIcon className="size-3.5 shrink-0 text-neutral-400" />
          <Slider
            value={[zoom]}
            onValueChange={(v) => setZoom(typeof v === 'number' ? v : v[0])}
            min={ZOOM_MIN}
            max={ZOOM_MAX}
            step={0.1}
            aria-label="Camera zoom"
            disabled={!!err}
            className="max-w-36"
          />
          <span className="w-8 shrink-0 font-mono text-[11px] text-neutral-400 tabular-nums">
            {zoom.toFixed(1)}x
          </span>
        </div>
        <button
          type="button"
          onClick={capture}
          disabled={!ready || !!err}
          aria-label="Take photo"
          title="Take photo"
          className="size-9 shrink-0 rounded-full border-2 border-white/80 bg-white/10 transition-colors hover:bg-white/30 disabled:opacity-40"
        />
        <div className="flex-1" />
      </div>
    </div>
  )
}
