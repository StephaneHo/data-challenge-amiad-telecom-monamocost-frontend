import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { useMutation } from '@tanstack/react-query'
import { searchRag } from '../api/rag'

const schema = z.object({
    query: z.string().min(3, 'Minimum 3 caractères'),
})

type FormData = z.infer<typeof schema>

export function useRagSearch() {
    const { register, handleSubmit, formState: { errors } } = useForm<FormData>({
        resolver: zodResolver(schema)
    })

    const mutation = useMutation({
        mutationFn: searchRag,
    })

    const onSubmit = handleSubmit((data: FormData) =>
        mutation.mutate(data))

    return { register, onSubmit, errors, mutation }


}