import React, { useState } from "react";
import { View, Text, Button, ScrollView, StyleSheet } from "react-native";
import axios from "axios";

export default function HttpScreen() {
  const [output, setOutput] = useState("");

  async function sendRequest() {
    try {
      const response = await axios.get("http://lk.stu.lipetsk.ru/", {
        headers: {
          "User-Agent": "ReactNativeApp/1.0",
          Accept: "text/html",
        },
      });
      setOutput(response.data);
    } catch (e: any) {
      setOutput("Error: " + e.message);
    }
  }

  return (
    <View style={styles.container}>
      <Button title="Отправить HTTP-запрос" onPress={sendRequest} />
      <ScrollView style={styles.output}>
        <Text>{output}</Text>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  output: { marginTop: 10 },
});
