import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet, FlatList } from "react-native";
import { Accelerometer, Gyroscope } from "expo-sensors";

const cStandards = [
  "C89 / ANSI C",
  "C90 / ISO C90",
  "C99 / ISO C99",
  "C11 / ISO C11",
  "C17 / ISO C17",
  "C23 / ISO C23",
];

export default function Home() {
  return (
    <View style={styles.container}>
      {/* Список стандартов C */}
      <Text style={styles.header}>C Standards</Text>
      <FlatList
        data={cStandards}
        keyExtractor={(item) => item}
        renderItem={({ item }) => <Text style={styles.listItem}>• {item}</Text>}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  header: { marginTop: 20, fontWeight: "bold", fontSize: 16 },
  listItem: { marginVertical: 2, fontSize: 14 },
});
