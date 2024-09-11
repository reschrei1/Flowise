import { ICommonObject, IFileUpload } from './Interface'
import { getCredentialData } from './utils'
import { type ClientOptions, OpenAIClient, toFile } from '@langchain/openai'
import { AssemblyAI } from 'assemblyai'
import { getFileFromStorage } from './storageUtils'
import fetch, { Headers } from 'node-fetch'

const SpeechToTextType = {
    OPENAI_WHISPER: 'openAIWhisper',
    ASSEMBLYAI_TRANSCRIBE: 'assemblyAiTranscribe',
    LOCALAI_STT: 'localAISTT'
}

export const convertSpeechToText = async (upload: IFileUpload, speechToTextConfig: ICommonObject, options: ICommonObject) => {
    if (speechToTextConfig) {
        const credentialId = speechToTextConfig.credentialId as string
        const credentialData = await getCredentialData(credentialId ?? '', options)
        const audio_file = await getFileFromStorage(upload.name, options.chatflowid, options.chatId)

        switch (speechToTextConfig.name) {
            case SpeechToTextType.OPENAI_WHISPER: {
                const url = 'https://aitfopenaiserv.openai.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01'
                const file = await toFile(audio_file, upload.name)
                const blob = await file.text()

                const formdata = new FormData()
                formdata.append('file', new Blob([blob]), 'file')
                formdata.append('prompt', '<string>')
                formdata.append('response_format', 'text')
                formdata.append('temperature', '0')

                try {
                    const response = await fetch(url, {
                        method: 'POST',
                        body: JSON.stringify(formdata),
                        headers: {
                            'Content-Type': 'multipart/form-data',
                            Accept: 'application/json',
                            'api-key': 'f9900f5517134606b0448ef9835f13a5'
                        }
                    })
                    const text = await response.text()
                    return text
                } catch (error) {
                    console.error(error)
                    // return ''
                }

                const azureOpenAIClientOptions: ClientOptions = {
                    apiKey: credentialData.azureOpenAIApiKey,
                    baseURL: 'https://aitfopenaiserv.openai.azure.com/openai/deployments/whisper/'
                    // endpoint: credentialData.azureOpenAIApiInstanceName,
                    // deployment: credentialData.azureOpenAIApiDeploymentName,
                    // apiVersion: credentialData.azureOpenAIApiVersion
                }
                const openAIClient = new OpenAIClient(azureOpenAIClientOptions)
                // const file = await toFile(audio_file, upload.name)
                const openAITranscription = await openAIClient.audio.transcriptions.create({
                    file: file,
                    model: 'whisper',
                    language: speechToTextConfig?.language,
                    temperature: speechToTextConfig?.temperature ? parseFloat(speechToTextConfig.temperature) : undefined,
                    prompt: speechToTextConfig?.prompt
                })
                if (openAITranscription?.text) {
                    return openAITranscription.text
                }
                break
            }
            case SpeechToTextType.ASSEMBLYAI_TRANSCRIBE: {
                const assemblyAIClient = new AssemblyAI({
                    apiKey: credentialData.assemblyAIApiKey
                })

                const params = {
                    audio: audio_file,
                    speaker_labels: false
                }

                const assemblyAITranscription = await assemblyAIClient.transcripts.transcribe(params)
                if (assemblyAITranscription?.text) {
                    return assemblyAITranscription.text
                }
                break
            }
            case SpeechToTextType.LOCALAI_STT: {
                const LocalAIClientOptions: ClientOptions = {
                    apiKey: credentialData.localAIApiKey,
                    baseURL: speechToTextConfig?.baseUrl
                }
                const localAIClient = new OpenAIClient(LocalAIClientOptions)
                const file = await toFile(audio_file, upload.name)
                const localAITranscription = await localAIClient.audio.transcriptions.create({
                    file: file,
                    model: speechToTextConfig?.model || 'whisper-1',
                    language: speechToTextConfig?.language,
                    temperature: speechToTextConfig?.temperature ? parseFloat(speechToTextConfig.temperature) : undefined,
                    prompt: speechToTextConfig?.prompt
                })
                if (localAITranscription?.text) {
                    return localAITranscription.text
                }
                break
            }
        }
    } else {
        throw new Error('Speech to text is not selected, but found a recorded audio file. Please fix the chain.')
    }
    return undefined
}
