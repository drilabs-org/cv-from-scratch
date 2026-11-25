import { useEffect, useRef } from 'react'
import { mountAsync, type Mounted } from 'pele/lazy'

export function MermaidBlock({ source }: { source: string }) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    // mountAsync() renders parse errors inside the element itself, no throw
    let mounted: Mounted | undefined
    let cancelled = false
    mountAsync(el, source, { zoom: { controls: false } }).then((m) => {
      if (cancelled) {
        m.destroy()
        return
      }
      mounted = m
    })
    return () => {
      cancelled = true
      mounted?.destroy()
    }
  }, [source])

  return <div ref={ref} className="mermaid-diagram" />
}
