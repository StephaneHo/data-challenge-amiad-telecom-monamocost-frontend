
import { Link } from 'react-router'
import { useRagSearch } from '../hooks/useRagSearch'
import { RagAnswer } from './RagAnswer'

export default function SearchPage() {

  const { register, onSubmit, errors, mutation } = useRagSearch()


  return (
    <>
      <form onSubmit={onSubmit}>
        <label htmlFor="search-query" className='block mb-1 text-sm font-medium'>Votre question</label>
        <input
          id="search-query"
          {...register('query')}
          placeholder='Votre question...'
          className='border p-2 rounded'
        />
        {errors.query && <p className="text-red-500 text-sm">{errors.query.message}</p>}
        <button
          type="submit"
          className='bg-blue-500 text-white px-4 py-2 rounded'
          disabled={mutation.isPending}>
          {mutation.isPending ? 'Recherche...' : 'Rechercher'}
        </button>
      </form>
      {mutation.isPending && <p>Recherche en cours...</p>}
      {mutation.isError && <p className="text-red-500">{mutation.error.message}</p>}
      {mutation.data && (
        <>
          <RagAnswer answer={mutation.data.answer} />
          <h2>Références</h2>
          {mutation.data.references.map((reference) => (
            <div key={reference.arxiv_id}>
              <Link to={`/papers/${reference.arxiv_id}`}>{reference.title}</Link>
            </div>
          ))}
        </>
      )
      }
    </>
  )
}

