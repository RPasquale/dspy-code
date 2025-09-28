import { useState } from 'react'

export default function TopicMultiSelect({ value, onChange, placeholder = 'spark.app,ui.action,training.dataset' }: { value: string[]; onChange: (topics: string[]) => void; placeholder?: string }) {
  const [input, setInput] = useState(value.join(','))
  return (
    <div className="flex items-center space-x-2">
      <input className="input w-96" value={input} onChange={e=> setInput(e.target.value)} placeholder={placeholder} />
      <button className="btn-primary" onClick={()=> onChange(input.split(',').map(s=>s.trim()).filter(Boolean))}>Apply</button>
    </div>
  )
}

