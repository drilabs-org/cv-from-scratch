import { useRef, useState } from 'react'
import { CameraIcon, ImagePlusIcon } from 'lucide-react'
import { ImageViewer } from '@/components/image-viewer'
import { EXAMPLES, InputActions } from '@/components/input-actions'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface Props {
  previewUrl: string | null
  fileName?: string | null
  onFile: (f: File) => void
  onClear?: () => void
  onOpenCamera?: () => void
  className?: string
  zoom?: number | null
  onZoomChange?: (zoom: number | null) => void
}

export function DropZone({ previewUrl, fileName, onFile, onClear, onOpenCamera, className, zoom, onZoomChange }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [loadingExample, setLoadingExample] = useState<string | null>(null)

  const loadExample = async (example: (typeof EXAMPLES)[number]) => {
    setLoadingExample(example.url)
    try {
      const res = await fetch(example.url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      onFile(new File([blob], example.name, { type: blob.type || 'image/png' }))
    } catch (e) {
      console.error('failed to load example image', e)
    } finally {
      setLoadingExample(null)
    }
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        const f = e.dataTransfer.files[0]
        if (f) onFile(f)
      }}
      className={cn(
        'relative flex h-72 cursor-pointer items-center justify-center overflow-hidden rounded-xl transition-colors',
        'focus-visible:outline-2 focus-visible:outline-ring',
        !previewUrl &&
          cn(
            'border-2 border-dashed border-border p-1.5',
            'hover:border-primary/50 hover:bg-accent/50',
            dragging && 'border-primary bg-accent',
          ),
        className,
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) onFile(f)
          e.target.value = ''
        }}
      />
      {previewUrl ? (
        <div
          className="h-full w-full cursor-default"
          onClick={(e) => e.stopPropagation()}
          onKeyDown={(e) => e.stopPropagation()}
        >
          {dragging && (
            <div className="pointer-events-none absolute inset-0 z-10 rounded-xl bg-accent/40 outline-2 -outline-offset-2 outline-dashed outline-primary" />
          )}
          <ImageViewer
            src={previewUrl}
            alt={fileName ?? 'input image'}
            className="h-full"
            zoom={zoom}
            onZoomChange={onZoomChange}
            actions={
              <InputActions
                onFile={onFile}
                onClear={onClear}
                onOpenCamera={onOpenCamera}
              />
            }
          />
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3 text-center">
          <div
            className={cn(
              'grid size-12 place-items-center rounded-xl border bg-muted/60 text-muted-foreground transition-colors',
              dragging && 'border-primary/40 bg-background text-primary',
            )}
          >
            <ImagePlusIcon className="size-5" />
          </div>
          <div className="flex flex-col gap-1">
            <p className="text-sm font-medium text-foreground">
              {dragging ? 'Release to load image' : 'Drop an image here'}
            </p>
            <p className="text-xs text-muted-foreground">PNG, JPG, WEBP</p>
          </div>
          <div className="flex items-center gap-2">
            {onOpenCamera && (
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={(e) => {
                  e.stopPropagation()
                  onOpenCamera()
                }}
                onKeyDown={(e) => e.stopPropagation()}
              >
                <CameraIcon data-icon="inline-start" />
                Open camera
              </Button>
            )}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                inputRef.current?.click()
              }}
              onKeyDown={(e) => e.stopPropagation()}
            >
              <ImagePlusIcon data-icon="inline-start" />
              Choose photo
            </Button>
          </div>
          <div className="flex flex-col items-center gap-2">
            <p className="text-xs text-muted-foreground">or try an example</p>
            <div className="flex items-center gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.url}
                  type="button"
                  disabled={loadingExample !== null}
                  title={ex.name}
                  onClick={(e) => {
                    e.stopPropagation()
                    void loadExample(ex)
                  }}
                  onKeyDown={(e) => e.stopPropagation()}
                  className={cn(
                    'size-14 overflow-hidden rounded-lg border border-border transition-all',
                    'hover:border-primary/60 hover:ring-2 hover:ring-primary/20',
                    'focus-visible:outline-2 focus-visible:outline-ring',
                    'disabled:cursor-wait disabled:opacity-50',
                    loadingExample === ex.url && 'border-primary ring-2 ring-primary/30',
                  )}
                >
                  <img
                    src={ex.url}
                    alt={ex.name}
                    className="size-full object-cover"
                  />
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
