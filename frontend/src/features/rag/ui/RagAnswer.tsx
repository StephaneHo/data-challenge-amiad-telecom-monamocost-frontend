import ReactMarkdown from 'react-markdown'

type RagAnswerProps = {
    answer: string
}

export function RagAnswer({ answer }: RagAnswerProps) {
    return (
        <div className="prose prose-sm max-w-none">
            <ReactMarkdown>{answer}</ReactMarkdown>
        </div>
    )
}