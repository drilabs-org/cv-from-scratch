export interface ParamSpec {
  name: string
  label?: string
  type: 'int' | 'float' | 'select'
  min?: number
  max?: number
  step?: number
  default?: number | string
  options?: string[]
  group?: string
}

export interface TaskMeta {
  id: string
  name: string
  project: string
  description: string
  params: ParamSpec[]
  input: string
  output: string
  result_tab?: 'auto' | 'input' | 'compare' | 'output'
  docs: Record<string, string>
}

export interface TaskResult {
  image: string | null
  data: Record<string, unknown>
}
