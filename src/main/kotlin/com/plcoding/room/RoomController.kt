package com.plcoding.room

import com.plcoding.data.MessageDataSource
import com.plcoding.data.model.Message
import io.ktor.http.cio.websocket.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.concurrent.ConcurrentHashMap
import nltk.stem.SnowballStemmer



class RoomController(
    private val messageDataSource: MessageDataSource
) {
    private val members = ConcurrentHashMap<String, Member>()

    fun onJoin(
        username: String,
        sessionId: String,
        socket: WebSocketSession
    ) {
        if(members.containsKey(username)) {
            throw MemberAlreadyExistsException()
        }
        members[username] = Member(
            username = username,
            sessionId = sessionId,
            socket = socket
        )
    }

    suspend fun sendMessage(senderUsername: String, message: String) {
        val sender = members[senderUsername]
            ?: throw Exception("Sender not found in members list.")
    
        val tokenizedMessage = tokenizeMessage(message)
        val stemmedMessage = stemWords(tokenizedMessage)
    
        // Keeping track of sender's previous messages to use in matching
        sender.previousMessages.add(stemmedMessage)
    
        val scores = mutableListOf<Pair<Member, Double>>()
        members.values.forEach { receiver ->
            val combinedInterests = receiver.interests + receiver.previousMessages
            val tfidf = calculateTFIDF(stemmedMessage, combinedInterests)
            val cosineSimilarity = cosineSimilarity(tfidf.first, tfidf.second)
            scores.add(Pair(receiver, cosineSimilarity))
        }
    

        val bestMatch = scores.maxBy { it.second }?.first
        ?: throw Exception("No best match found.")

        // Send the message to the best match
        val incomingMessage = Message(
            text = message,
            username = senderUsername,
            timestamp = System.currentTimeMillis()
        )
        messageDataSource.insertMessage(incomingMessage)

        val parsedMessage = Json.encodeToString(incomingMessage)
        bestMatch.socket.send(Frame.Text(parsedMessage))
    }

    fun calculateTFIDF(stemmedMessage: List<String>, combinedInterests: List<String>): Pair<Map<String, Double>, Map<String, Double>> {
        val messageTf = calculateTf(stemmedMessage)
        val interestsTf = calculateTf(combinedInterests)
        val idf = calculateIdf(listOf(stemmedMessage, combinedInterests))
        val messageTfidf = multiplyTfIdf(messageTf, idf)
        val interestsTfidf = multiplyTfIdf(interestsTf, idf)
        return Pair(messageTfidf, interestsTfidf)
    }
    
    private fun calculateTf(words: List<String>): Map<String, Double> {
        val wordCounts = words.groupBy { it }.mapValues { it.value.count().toDouble() }
        val totalWordCount = wordCounts.values.sum()
        return wordCounts.mapValues { it.value / totalWordCount }
    }
    
    private fun calculateIdf(lists: List<List<String>>): Map<String, Double> {
        val wordDocFrequency = mutableMapOf<String, Double>()
        for (words in lists) {
            for (word in words.distinct()) {
                wordDocFrequency[word] = wordDocFrequency.getOrDefault(word, 0.0) + 1.0
            }
        }
        val totalDocCount = lists.size.toDouble()
        return wordDocFrequency.mapValues { Math.log(totalDocCount / it.value) }
    }
    
    private fun multiplyTfIdf(tf: Map<String, Double>, idf: Map<String, Double>): Map<String, Double> {
        return tf.mapValues { it.value * idf[it.key]!! }
    }

    private fun tokenizeMessage(text: String): List<String> {
        val stopWords = setOf("the", "and", "a", "to", "of", "in", "that", "it", "with", "as", "for", "was", "on", "are", "at", "by", "be", "this", "which", "or", "an", "but", "not", "is", "are", "from", "was", "we", "they", "say", "will", "would", "can", "if", "has", "have", "had", "do", "does", "did", "its", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "one", "every", "least", "less", "many", "now", "ever", "never", "also", "may", "might", "must", "need", "ought", "shall", "should", "will", "would")
        return text.toLowerCase().split("\\W+").filter { it.isNotEmpty() && !stopWords.contains(it) }
    }
    

    private fun stemWords(words: List<String>): List<String> {
        val stemmer = SnowballStemmer("english")
        return words.map { stemmer.stem(it) }
    }


    private fun cosineSimilarity(vec1: DoubleArray, vec2: DoubleArray): Double {
        var dotProduct = 0.0
        var vec1Magnitude = 0.0
        var vec2Magnitude = 0.0
        for (i in vec1.indices) {
            dotProduct += vec1[i] * vec2[i]
            vec1Magnitude += vec1[i] * vec1[i]
            vec2Magnitude += vec2[i] * vec2[i]
        }
        return dotProduct / (sqrt(vec1Magnitude) * sqrt(vec2Magnitude))
    }


    suspend fun tryDisconnect(username: String) {
        members[username]?.socket?.close()
        if(members.containsKey(username)) {
            members.remove(username)
        }
    }
}