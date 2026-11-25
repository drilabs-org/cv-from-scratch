import {
  Contrast,
  Droplets,
  Grid2x2,
  Hash,
  ScanSearch,
  Spline,
  type LucideIcon,
} from 'lucide-react'
import { ScrollArea } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'
import type { TaskMeta } from '@/types'

interface Props {
  tasks: TaskMeta[]
  selectedId: string | null
  onSelect: (id: string) => void
}

const ICON_RULES: [RegExp, LucideIcon][] = [
  [/gray/i, Contrast],
  [/blur/i, Droplets],
  [/edge|canny/i, Spline],
  [/binary|thresh/i, Grid2x2],
  [/count|object/i, Hash],
]

function iconFor(task: TaskMeta): LucideIcon {
  for (const [re, icon] of ICON_RULES) {
    if (re.test(task.id) || re.test(task.name)) return icon
  }
  return ScanSearch
}

export function TaskSidebar({ tasks, selectedId, onSelect }: Props) {
  const groups = new Map<string, TaskMeta[]>()
  for (const t of tasks) {
    const list = groups.get(t.project) ?? []
    list.push(t)
    groups.set(t.project, list)
  }

  return (
    <ScrollArea className="h-full">
      <nav className="flex flex-col gap-5 p-3">
        {[...groups.entries()].map(([project, items]) => (
          <div key={project} className="flex flex-col gap-0.5">
            <p className="px-2 pb-1 font-mono text-[11px] tracking-wider text-muted-foreground uppercase">
              {project}
            </p>
            {items.map((t) => {
              const active = t.id === selectedId
              const Icon = iconFor(t)
              return (
                <button
                  key={t.id}
                  type="button"
                  title={t.description}
                  onClick={() => onSelect(t.id)}
                  className={cn(
                    'group flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm transition-colors',
                    active
                      ? 'bg-accent font-medium text-accent-foreground shadow-[inset_2px_0_0_var(--color-primary)]'
                      : 'text-muted-foreground hover:bg-accent/60 hover:text-foreground',
                  )}
                >
                  <Icon
                    className={cn(
                      'size-4 shrink-0 transition-colors',
                      active
                        ? 'text-primary'
                        : 'text-muted-foreground/60 group-hover:text-foreground',
                    )}
                  />
                  <span className="truncate">{t.name}</span>
                </button>
              )
            })}
          </div>
        ))}
      </nav>
    </ScrollArea>
  )
}
