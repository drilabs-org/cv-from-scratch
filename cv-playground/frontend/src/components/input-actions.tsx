import { useRef } from 'react'
import { CameraIcon, ImagePlusIcon, ShapesIcon, XIcon } from 'lucide-react'
import { ViewerToolButton } from '@/components/image-viewer'

export const EXAMPLES = [
  { url: '/examples/sample.png', name: 'sample.png' },
  { url: '/examples/sample2.png', name: 'sample2.png' },
  { url: '/examples/sample3.png', name: 'sample3.png' },
  { url: '/examples/coins.jpg', name: 'coins.jpg' },
  { url: '/examples/digit.png', name: 'digit.png' },
  { url: '/examples/digit-hard.png', name: 'digit-hard.png' },
  { url: '/examples/digit-photo.png', name: 'digit-photo.png' },
  { url: '/examples/digit-3.png', name: 'digit-3.png' },
  { url: '/examples/digit-8.png', name: 'digit-8.png' },
]

interface Props {
  onFile: (f: File) => void
  onClear?: () => void
  onOpenCamera?: () => void
}

export function InputActions({ onFile, onClear, onOpenCamera }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  const loadExample = async () => {
    const example = EXAMPLES[0]
    try {
      const res = await fetch(example.url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      onFile(new File([blob], example.name, { type: blob.type || 'image/png' }))
    } catch (e) {
      console.error('failed to load example image', e)
    }
  }

  return (
    <>
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
      <ViewerToolButton
        label="Load example image"
        onClick={() => void loadExample()}
      >
        <ShapesIcon />
      </ViewerToolButton>
      {onOpenCamera && (
        <ViewerToolButton label="Take a photo" onClick={onOpenCamera}>
          <CameraIcon />
        </ViewerToolButton>
      )}
      <ViewerToolButton
        label="Change image"
        onClick={() => inputRef.current?.click()}
      >
        <ImagePlusIcon />
      </ViewerToolButton>
      {onClear && (
        <ViewerToolButton label="Clear image" onClick={onClear}>
          <XIcon />
        </ViewerToolButton>
      )}
    </>
  )
}
