---
layout: single
title: "close_loop/github_io"
categories : Control 
tag: [control, loop]
toc: true
author_profile : false
sidebar:
     nav: "docs/Control"
---



# 🚗 Control Systems Lectures - Closed Loop Control

## 🎯 Introduction

This lecture covers **open-loop** vs **closed-loop** control systems.

> A **control system** is a mechanism that alters the future behavior or state of a system. For it to be considered a control system (not just a mechanism that changes a state), the output must **tend toward a desired state**.

**Control theory** is the mathematical strategy used to select appropriate inputs to achieve desired outputs. Without it, engineers would rely solely on trial and error.

## ⚙️ Components of a Control System

Every control system consists of two basic parts:

- **The Plant**: the system to be controlled
- **The Input**: acts on the plant and generates the system output

### 🔓 Open-Loop Control

In open-loop systems, the **input does not depend on the output**. These are used for simple, predictable processes.

#### ✅ Examples:

- **Dishwasher**: Runs for a preset **time** regardless of how clean the dishes actually are.
- **Sprinkler System**: Waters the lawn based on a **timer**, not soil moisture level.
- **Car Without Cruise Control**: Pedal is fixed, so speed varies with hills or valleys.

**Main Drawback**: Open-loop systems cannot adjust for disturbances or changes in the system.

### 🔁 Closed-Loop (Feedback) Control

In closed-loop systems, the **output is measured and fed back** to adjust the input.

Also known as:

- Feedback Control
- Negative Feedback
- Automatic Control

#### 🧠 How It Works:

1. Measure output using a sensor.
2. Compare output to the reference (desired) value.
3. Generate an error term.
4. Feed error into the controller.
5. Controller adjusts the input to reduce the error.

This creates a **feedback loop** where the system continually tries to drive the error to zero.

#### ✅ Examples:

- **Dishwasher with Cleanliness Sensor**: Stops when dishes are actually clean.
- **Sprinkler with Moisture Sensor**: Runs only until soil moisture reaches a set level.
- **Car with Cruise Control**:
  - Sensor = speedometer
  - Reference = desired speed (e.g., 100 mph)
  - If car slows on a hill, controller increases gas.
  - If car speeds up downhill, controller decreases gas.

## 📐 Block Diagram Abstraction

Let’s define abstract labels in a block diagram:

- `V` : Reference Signal
- `D` : Controller
- `G` : Plant
- `Y` : Output
- `H` : Sensor
- `E` : Error Term

From the diagram, we get the equation:

```
E = V - H * Y
Y = E * D * G
```

Solve these to find `Y` in terms of `V`:

```
Y = (D * G / (1 + D * G * H)) * V
```

This is the **transfer function** of the closed-loop system.

> 🎯 Insight: The feedback path alters the behavior of the original plant. The resulting system behaves like an open-loop system with a modified plant that now tracks the input better.

## ❓ Final Thought

> Can any plant `G` be made to behave however we want just by adding a controller `D` and sensor `H`?

For instance, in the car example:

- Can we turn a **Pinto into a Ferrari** just by applying more gas and using feedback?

We'll discuss that in the next lecture!







