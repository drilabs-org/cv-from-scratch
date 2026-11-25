import { Children, isValidElement, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { MermaidBlock } from '@/components/mermaid-block'
import { cn } from '@/lib/utils'
import 'github-markdown-css/github-markdown-light.css'
import 'highlight.js/styles/github.css'
import 'katex/dist/katex.min.css'

const LANG_LABELS: Record<string, string> = {
  en: 'English',
  vi: 'Tiếng Việt',
  spec: 'Spec',
}

function mermaidSource(node: ReactNode): string | null {
  for (const child of Children.toArray(node)) {
    if (!isValidElement(child)) continue
    const props = child.props as { className?: string; children?: ReactNode }
    if (/\blanguage-mermaid\b/.test(props.className ?? '')) {
      return String(props.children ?? '')
    }
  }
  return null
}

export function DocsPanel({ docs }: { docs: Record<string, string> }) {
  const langs = Object.keys(docs).filter((l) => docs[l]?.trim())
  const [lang, setLang] = useState('en')
  const active = langs.includes(lang) ? lang : langs[0]

  if (langs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        This task has no docs yet.
      </p>
    )
  }
  return (
    <div>
      {langs.length > 1 && (
        <div className="mb-4 inline-flex gap-1 rounded-lg border bg-muted/40 p-1">
          {langs.map((l) => (
            <button
              key={l}
              type="button"
              onClick={() => setLang(l)}
              className={cn(
                'rounded-md px-3 py-1 text-xs font-medium transition-colors',
                l === active
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground',
              )}
            >
              {LANG_LABELS[l] ?? l.toUpperCase()}
            </button>
          ))}
        </div>
      )}
      <div className="markdown-body">
        <ReactMarkdown
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[
            rehypeKatex,
            [rehypeHighlight, { plainText: ['mermaid'] }],
          ]}
          components={{
            pre({ children }) {
              const source = mermaidSource(children)
              return source === null ? (
                <pre>{children}</pre>
              ) : (
                <MermaidBlock source={source} />
              )
            },
          }}
        >
          {docs[active]}
        </ReactMarkdown>
      </div>
    </div>
  )
}
