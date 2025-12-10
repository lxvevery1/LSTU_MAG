import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';

export default function HomeScreen() {
    const [accel, setAccel] = useState({ x: 0, y: 0, z: 0 });
    const [gyro, setGyro] = useState({ x: 0, y: 0, z: 0 });

    useEffect(() => {
        Accelerometer.setUpdateInterval(100);
        Gyroscope.setUpdateInterval(100);

        const subA = Accelerometer.addListener(setAccel);
        const subG = Gyroscope.addListener(setGyro);

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

        <Text style={styles.header}>Gyroscope</Text>
        <Text>X: {gyro.x.toFixed(2)}</Text>
        <Text>Y: {gyro.y.toFixed(2)}</Text>
        <Text>Z: {gyro.z.toFixed(2)}</Text>
        </View>
    );
}

const styles = StyleSheet.create({
    container: { padding: 20 },
    header: { marginTop: 20, fontWeight: 'bold', fontSize: 16 }
});
