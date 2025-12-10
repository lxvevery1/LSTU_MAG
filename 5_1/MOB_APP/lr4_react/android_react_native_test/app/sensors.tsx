import React, { useEffect, useState } from "react";
import { View, Text, StyleSheet } from "react-native";
import { Accelerometer } from "expo-sensors";

export default function Home() {
  const [accel, setAccel] = useState({ x: 0, y: 0, z: 0 });
  const [gyro, setGyro] = useState({ x: 0, y: 0, z: 0 });

  useEffect(() => {
    Accelerometer.setUpdateInterval(100);

    const subA = Accelerometer.addListener(setAccel);

    return () => {
      subA.remove();
      subG.remove();
    };
  }, []);

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Accelerometer</Text>
      <Text>X: {accel.x.toFixed(2)}</Text>
      <Text>Y: {accel.y.toFixed(2)}</Text>
      <Text>Z: {accel.z.toFixed(2)}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20 },
  header: { marginTop: 20, fontWeight: "bold", fontSize: 16 },
});
