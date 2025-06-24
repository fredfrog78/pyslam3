## Specifications

```
v1.0 2023.
```
# LVX


```
Searching for Keywords
Search for keywords such as “battery” and “install” to find a topic. If you are using Adobe
Acrobat Reader to read this document, press Ctrl+F on Windows or Command+F on Mac
to begin a search.
```
```
Navigating to a Topic
View a complete list of topics in the table of contents. Click on a topic to navigate to that
section.
```
```
Printing this Document
This document supports high resolution printing.
```
This document is copyrighted by DJI with all rights reserved. Unless otherwise authorized by
DJI, you are not eligible to use or allow others to use the document or any part of the document
by reproducing, transferring or selling the document. Users should only refer to this document
and the content thereof as instructions to operate DJI UAV. The document should not be used
for other purposes.


## Contents

   - © 2023 Livox Tech. All Rights Reserved.
- LVX2 Format Definition
- Data Types
- Public Header Block
- Private Header Block
- Devices Info Block
- Point Cloud Data Block


LVX2 Specifications

4 © 2023 Livox Tech. All Rights Reserved.

This document describes the specifications of LVX2 format v1.0. The LVX2 file is a point cloud file
format developed by Livox Tech, based on the company's LiDAR sensors. This file format allows
users to play the point cloud file at a base frequency of 20 Hz. At the same time, users can also
acquire point data from a single device from this file for more complex algorithm development.

## LVX2 Format Definition

The format contains binary data consisting of public header block, private header block, device
info block, and point cloud data block.

## Public Header Block

### PRIVATE HEADER BLOCK

### DEVICE INFO BLOCK

### POINT DATA BLOCK

All data are in little-endian format. The header block consists of file signature, version
information, and a magic code. The length of the device info block is variable, capable of
accommodating any number of devices. The point cloud data block has point cloud data
organized by package, and these packages are organized by frames in each file.

## Data Types

The following data types are used in the LVX2 format.

- char (1 byte)
- unsigned char (1 byte)
- short (2 bytes)
- unsigned short (2 bytes)
- int (4 bytes)
- unsigned int (4 bytes)
- long long (8 bytes)
- unsigned long long (8 bytes)
- float (4 bytes IEEE floating point format)
- double (8 bytes IEEE floating point format)

## Public Header Block

```
Item Format Size
File Signature (“livox_tech”) char[16] 16 bytes
Version-A char 1 bytes
Version-B char 1 bytes
Version-C char 1 bytes
Version-D char 1 bytes
Magic Code unsigned int 4 bytes
```

```
LVX2 Specifications
```
```
© 2023 Livox Tech. All Rights Reserved. 5
```
**File Signature:** The file signature must contain “livox_tech” as it is required by the LVX
specification. These characters can be checked by Livox Viewer as an initial determination of
file type. Note that the first 10 bytes should be “livox_tech”, and the last 6 bytes should be zero
filled.
**Version:** Version a is 2. Version b is 0. Version c is 0. Version d is 0.
**Magic Code:** This field should be a value of 0xAC0EA767. Livox Viewer will not identify a LVX
file with an incorrect Magic Code.

## Private Header Block

```
Item Format Size
Frame Duration unsigned int 4 bytes
Device Count unsigned char 1 byte
```
**Frame Duration:** The duration of one frame. The unit of duration is millisecond (ms). Note:
This field is only used to inform the user of the frame duration of the current file. In the 2.0.0.
version of the LVX2 file, this field is 50 and cannot be changed.
**Device Count:** The count of device info block is variable to suit several devices. This field should
be a value of the count of devices.

## Devices Info Block

```
Item Format Size
Device Info 0 struct 63 bytes
......
Device Info N struct 63 bytes
```
- N = Device Count - 1. Device Count is inside Private Header Block.

**Device Info:** This is a field that provides information of each device. This field is defined as:

```
Item Format Size Description
LiDAR SN Code char[16] 16 bytes LiDAR broadcast code
Hub SN Code char[16] 16 bytes Hub broadcast code.
Note that an empty hub SN means there
is no hub connecting this LiDAR.
LiDAR_ID unsigned int 4 byte LiDAR ID used to identify the device.
LiDAR_Type unsigned char 1 byte Reserved field
Device_Type unsigned char 1 byte Device Type:
9: Mid-
10: HAP
```

LVX2 Specifications

6 © 2023 Livox Tech. All Rights Reserved.

```
Extrinsic Enable unsigned char 1 byte 0: Extrinsic parameters disabled. Cloud
points should be computed without extrinsic
parameters.
1: Extrinsic parameters enabled. Cloud
points should be computed with extrinsic
parameters.
Roll float 4 bytes Extrinsic parameters:
Roll Angle, Unit: degree
Pitch float 4 bytes Extrinsic parameters:
Pitch Angle, Unit: degree
Yaw float 4 bytes Extrinsic parameters:
Yaw Angle, Unit: degree
X float 4 bytes Extrinsic parameters:
X Translation, Unit: m
Y float 4 bytes Extrinsic parameters:
Y Translation, Unit: m
Z float 4 bytes Extrinsic parameters:
Z Translation, Unit: m
```
- Users can use LiDAR ID to extract point cloud data of each device from a LVX2 file.

## Point Cloud Data Block

Data from the Point Cloud Data Block are composed of frames, and each frame is composed of
packages.

```
Item Format Size
Frame 0 struct N bytes (N = Next Offset – Current Offset)
Frame 1 struct N bytes
......
Frame N struct N bytes
```
Frame is defined as:

```
Item Format Size
Frame Header struct 24 bytes
Package 0 struct Depends on current package header.
Package 1 struct Depends on current package header.
......
Package N struct Depends on current package header.
```

```
LVX2 Specifications
```
```
© 2023 Livox Tech. All Rights Reserved. 7
```
Frame Header is defined as:

```
Item Format Size Description
Current Offset long long 8 bytes Absolute offset of the current frame in this file.
Next Offset long long 8 bytes Absolute offset of next frame in this file.
Frame Index long long 8 bytes Current Frame Index.
```
Package is defined as SDK protocol normal data:

```
Item Format Size Description
Version unsigned char 1 byte Package protocol version, 0 for the current
version
LiDAR ID unsigned int 4 byte LiDAR ID used to identify the device.
LiDAR_Type unsigned char 1 byte Reserved field
```
```
Timestamp Type unsigned char 1 byte
```
```
Refer to Livox SDK2 Communication Protocol
for more details.
0x00: if there is no time synchronization, use
LiDAR system time.
0x01: gPTP/PTP time synchronization
0x02: GPS time synchronization
Timestamp unsiged char[8] 8 bytes Nanosecond of the point package time depends
on Timestamp Type.
Udp Counter unsigned short 2 bytes The udp counter index inside current udp
packet data from Ethernet.
Data Type unsigned char 1 byte Point Cloud Coordinate Format:
1: Point Cloud Data
2: Point Cloud Data
```
```
Length unsigned int 4 byte
```
```
The length of the point cloud data of this
package (excluding the package header part
and starting length calculation from Point 0 to
Point N).
For Data Type 0x01, size per point is 14 bytes.
For Data Type 0x02, size per point is 8 bytes.
Therefore the point count inside this package is
Length divided by size per point.
Frame_Counter unsiged char 1 bytes Reserved field
Reserve unsiged char[4] 4 bytes Reserved bytes
Point 0 struct depends Point information, depending on Data Type
Point 1 struct depends Point information, depending on Data Type
......
Point N struct depends Point information, depending on Data Type
```

LVX2 Specifications

Point is defined as follows:
Data type 0x01 (14 bytes per point)

```
Item Format Size Description
x int 4 bytes X-axis position, unit: mm
y int 4 bytes Y-axis position, unit: mm
z int 4 bytes Z-axis position, unit: mm
reflectivity unsigned char 1 byte Reflectivity
tag unsigned char 1 byte Refer to Livox SDK2 Communication Protocol for
more details.
```
Data type 0x02 (8 bytes per point)

```
Item Format Size Description
x short 2 bytes X-axis position, unit: cm
y short 2 bytes Y-axis position, unit: cm
z short 2 bytes Z-axis position, unit: cm
reflectivity unsigned char 1 byte Reflectivity
tag unsigned char 1 byte Refer to Livox SDK2 Communication Protocol for
more details.
```
- The point count inside one package is based on the package length and the size per
    point. The size per point is based on the point data type (such as Data type 0x01 with
    14 bytes per point and Data type 0x02 with 8 bytes per point). For HAP and Mid-
    LiDAR, the point count of each package is 96.
- For data type 0x01, the unit of the x, y, z is mm. For data type 0x02, the unit of the x, y, z
    is cm.
- For more details about Livox SDK2 Communication Protocol, go to https://github.com/
    Livox-SDK/Livox-SDK2.

Copyright © 2023 Livox Tech. All Rights Reserved.
Livox and Livox Mid are trademarks of Livox Technology Company Limited.



