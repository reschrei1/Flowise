import { END } from '@langchain/langgraph'
import { INode, INodeData, INodeParams, ISeqAgentNode } from '../../../src/Interface'

class End_SeqAgents implements INode {
    label: string
    name: string
    version: number
    description: string
    type: string
    icon: string
    category: string
    baseClasses: string[]
    credential: INodeParams
    inputs: INodeParams[]
    hideOutput: boolean

    constructor() {
        this.label = 'End'
        this.name = 'seqEnd'
        this.version = 1.0
        this.type = 'End'
        this.icon = 'end.svg'
        this.category = 'Sequential Agents'
        this.description = 'End conversation'
        this.baseClasses = [this.type]
        this.inputs = [
            {
                label: 'Start | Agent | LLM | Tool Node',
                name: 'sequentialNode',
                type: 'Start | Agent | LLMNode | ToolNode'
            }
        ]
        this.hideOutput = true
    }

    async init(nodeData: INodeData): Promise<any> {
        const sequentialNode = nodeData.inputs?.sequentialNode as ISeqAgentNode
        if (!sequentialNode) throw new Error('End must have a predecessor!')

        const returnOutput: ISeqAgentNode = {
            id: nodeData.id,
            node: END,
            name: END,
            label: END,
            type: 'end',
            output: END,
            predecessorAgents: [sequentialNode]
        }

        return returnOutput
    }
}

module.exports = { nodeClass: End_SeqAgents }
