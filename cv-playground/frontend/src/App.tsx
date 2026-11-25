import { useEffect, useMemo, useState } from 'react'
import {
  CornerDownLeftIcon,
  PlayIcon,
  RotateCcwIcon,
  ScanEyeIcon,
  TriangleAlertIcon,
} from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { Spinner } from '@/components/ui/spinner'
import { DocsPanel } from '@/components/docs-panel'
import {
  ParamsForm,
  defaultsFor,
  type ParamValues,
} from '@/components/params-form'
import { ResultView } from '@/components/result-view'
import { TaskSidebar } from '@/components/task-sidebar'
import { ThemeToggle } from '@/components/theme-toggle'
import type { TaskMeta, TaskResult } from '@/types'

interface TaskState {
  params: ParamValues
  result: TaskResult | null
  elapsedMs: number | null
  error: string | null
}

function newTaskState(t: TaskMeta): TaskState {
  return {
    params: defaultsFor(t.params),
    result: null,
    elapsedMs: null,
    error: null,
  }
}

export default function App() {
  const [tasks, setTasks] = useState<TaskMeta[] | null>(null)
  const [tasksError, setTasksError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [taskStates, setTaskStates] = useState<Record<string, TaskState>>({})
  const [loadingId, setLoadingId] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/tasks')
      .then((r) => {
        if (!r.ok) throw new Error(`GET /api/tasks -> HTTP ${r.status}`)
        return r.json()
      })
      .then((list: TaskMeta[]) => {
        setTasks(list)
        setSelectedId(list[0]?.id ?? null)
      })
      .catch((e) => setTasksError(e instanceof Error ? e.message : String(e)))
  }, [])

  const task = useMemo(
    () => tasks?.find((t) => t.id === selectedId) ?? null,
    [tasks, selectedId],
  )

  const ts = task ? (taskStates[task.id] ?? newTaskState(task)) : null
  const loading = task !== null && loadingId === task.id

  const patchTask = (t: TaskMeta, patch: Partial<TaskState>) => {
    setTaskStates((prev) => ({
      ...prev,
      [t.id]: { ...(prev[t.id] ?? newTaskState(t)), ...patch },
    }))
  }

  const clearAllOutputs = () => {
    setTaskStates((prev) =>
      Object.fromEntries(
        Object.entries(prev).map(([id, s]) => [
          id,
          { ...s, result: null, elapsedMs: null, error: null },
        ]),
      ),
    )
  }

  const selectTask = (id: string) => {
    setSelectedId(id)
  }

  const handleFile = (f: File) => {
    setFile(f)
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(f)
    })
    clearAllOutputs()
  }

  const clearImage = () => {
    setFile(null)
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old)
      return null
    })
    clearAllOutputs()
  }

  const run = async () => {
    if (!task || !file || !ts) return
    const t = task
    setLoadingId(t.id)
    patchTask(t, { error: null })
    const t0 = performance.now()
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('params', JSON.stringify(ts.params))
      const res = await fetch(`/api/tasks/${t.id}`, {
        method: 'POST',
        body: fd,
      })
      const body = await res.json().catch(() => null)
      if (!res.ok) {
        throw new Error(body?.detail ?? `HTTP ${res.status}`)
      }
      patchTask(t, { result: body, elapsedMs: performance.now() - t0 })
    } catch (e) {
      patchTask(t, { error: e instanceof Error ? e.message : String(e) })
    } finally {
      setLoadingId((cur) => (cur === t.id ? null : cur))
    }
  }

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        void run()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const isMac = navigator.platform.toUpperCase().includes('MAC')

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b px-4">
        <div className="grid size-8 shrink-0 place-items-center rounded-lg bg-primary text-primary-foreground">
          <ScanEyeIcon className="size-5" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-sm leading-tight font-semibold tracking-tight">
            CV Playground
          </h1>
          <span className="text-[11px] leading-tight text-muted-foreground">
            Image processing lab
          </span>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {task && (
            <Badge variant="outline" className="font-mono text-[11px]">
              {task.project}
            </Badge>
          )}
          <ThemeToggle />
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="w-64 shrink-0 border-r bg-muted/30">
          {tasks ? (
            <TaskSidebar
              tasks={tasks}
              selectedId={selectedId}
              onSelect={selectTask}
            />
          ) : (
            <div className="flex flex-col gap-2 p-3">
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
              <Skeleton className="h-8 w-full" />
            </div>
          )}
        </aside>

        <main className="min-w-0 flex-1 overflow-y-auto p-5">
          <div className="mx-auto w-full max-w-6xl">
          {tasksError && (
            <Alert variant="destructive" className="mb-4">
              <TriangleAlertIcon />
              <AlertTitle>Failed to load task list</AlertTitle>
              <AlertDescription>{tasksError}</AlertDescription>
            </Alert>
          )}

          {task && ts && (
            <div className="flex flex-col gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>{task.name}</CardTitle>
                  <CardDescription>{task.description}</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-4">
                  <ResultView
                    result={ts.result}
                    loading={loading}
                    inputUrl={previewUrl}
                    fileName={file?.name}
                    onFile={handleFile}
                    onClear={clearImage}
                    elapsedMs={ts.elapsedMs}
                    preferredTab={task.result_tab}
                  />
                  <ParamsForm
                    params={task.params}
                    values={ts.params}
                    onChange={(name, v) =>
                      patchTask(task, {
                        params: { ...ts.params, [name]: v },
                      })
                    }
                  />
                  <div className="flex items-center gap-3">
                    <Button onClick={run} disabled={!file || loading}>
                      {loading ? (
                        <Spinner data-icon="inline-start" />
                      ) : (
                        <PlayIcon data-icon="inline-start" />
                      )}
                      Run task
                    </Button>
                    {task.params.length > 0 && (
                      <Button
                        variant="outline"
                        onClick={() =>
                          patchTask(task, {
                            params: defaultsFor(task.params),
                          })
                        }
                      >
                        <RotateCcwIcon data-icon="inline-start" />
                        Reset
                      </Button>
                    )}
                    <kbd className="hidden items-center gap-1 rounded-md border bg-muted px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground sm:flex">
                      {isMac ? 'Cmd' : 'Ctrl'} + Enter
                    </kbd>
                  </div>
                  {ts.error && (
                    <Alert variant="destructive">
                      <TriangleAlertIcon />
                      <AlertTitle>Error while running task</AlertTitle>
                      <AlertDescription>{ts.error}</AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Docs</CardTitle>
                  <CardDescription>
                    Concept explanation for the selected task.
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <DocsPanel docs={task.docs} />
                </CardContent>
              </Card>
            </div>
          )}
          </div>
        </main>
      </div>

      <footer className="flex h-8 shrink-0 items-center gap-4 border-t px-4 font-mono text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className={
              tasks ? 'size-1.5 rounded-full bg-primary' : 'size-1.5 animate-pulse rounded-full bg-muted-foreground/40'
            }
          />
          {tasks ? `${tasks.length} tasks` : 'connecting'}
        </span>
        {task && <span className="truncate">{task.id}</span>}
        <span className="ml-auto hidden items-center gap-1 sm:flex">
          {isMac ? 'Cmd' : 'Ctrl'}
          <CornerDownLeftIcon className="size-3" />
          to run
        </span>
      </footer>
    </div>
  )
}
