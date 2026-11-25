import { Collapsible } from '@base-ui/react/collapsible'
import { ChevronDownIcon } from 'lucide-react'
import { ElasticSlider } from '@/components/elastic-slider'
import { Field, FieldGroup, FieldLabel } from '@/components/ui/field'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { ParamSpec } from '@/types'

export type ParamValues = Record<string, number | string>

export function defaultsFor(params: ParamSpec[]): ParamValues {
  return Object.fromEntries(
    params
      .filter((p) => p.default !== undefined)
      .map((p) => [p.name, p.default as number | string]),
  )
}

interface Props {
  params: ParamSpec[]
  values: ParamValues
  onChange: (name: string, v: number | string) => void
}

function ParamField({
  p,
  values,
  onChange,
}: {
  p: ParamSpec
  values: ParamValues
  onChange: Props['onChange']
}) {
  const value = values[p.name] ?? p.default

  if (p.type === 'select') {
    return (
      <Field>
        <FieldLabel>{p.label ?? p.name}</FieldLabel>
        <Select
          value={String(value ?? '')}
          onValueChange={(v) => onChange(p.name, String(v))}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {p.options?.map((o) => (
                <SelectItem key={o} value={o}>
                  {o}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </Field>
    )
  }

  return (
    <Field>
      <ElasticSlider
        className="[--elastic-slider-focus:var(--primary)] [--elastic-slider-handle:var(--primary)]"
        label={p.label ?? p.name}
        value={Number(value ?? p.min ?? 0)}
        min={p.min}
        max={p.max}
        step={p.step ?? (p.type === 'int' ? 1 : 0.1)}
        onValueChange={(v) => onChange(p.name, v)}
      />
    </Field>
  )
}

export function ParamsForm({ params, values, onChange }: Props) {
  if (!params.length) return null

  const main = params.filter((p) => !p.group)
  const groups = new Map<string, ParamSpec[]>()
  for (const p of params) {
    if (!p.group) continue
    const list = groups.get(p.group) ?? []
    list.push(p)
    groups.set(p.group, list)
  }

  return (
    <FieldGroup>
      {main.map((p) => (
        <ParamField key={p.name} p={p} values={values} onChange={onChange} />
      ))}

      {groups.size > 0 && (
        <Collapsible.Root className="group flex flex-col">
          <Collapsible.Trigger className="flex w-full items-center gap-1.5 border-t pt-3 text-xs font-medium text-muted-foreground hover:text-foreground">
            <ChevronDownIcon className="size-3.5 transition-transform group-data-[open]:rotate-180" />
            Advanced
          </Collapsible.Trigger>
          <Collapsible.Panel className="flex flex-col gap-4 pt-3">
            {[...groups].map(([group, ps]) => (
              <div key={group} className="flex flex-col gap-3">
                <div className="text-[11px] font-medium tracking-wide text-muted-foreground uppercase">
                  {group}
                </div>
                {ps.map((p) => (
                  <ParamField
                    key={p.name}
                    p={p}
                    values={values}
                    onChange={onChange}
                  />
                ))}
              </div>
            ))}
          </Collapsible.Panel>
        </Collapsible.Root>
      )}
    </FieldGroup>
  )
}
