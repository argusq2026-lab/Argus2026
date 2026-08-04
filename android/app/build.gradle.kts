plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.argus.edge"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.argus.edge"
        // Snapdragon 8 Elite (SM8750) ships Android 15. 31 is a safe floor that
        // still covers the CameraX APIs used here.
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // arm64-v8a only. The Hexagon NPU exists on no other ABI, and shipping
        // a 32-bit slice would produce a build that installs and then cannot
        // load the QNN backend -- exactly the quiet degradation this project
        // refuses everywhere else.
        ndk { abiFilters += "arm64-v8a" }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    sourceSets {
        getByName("main").java.srcDirs("src/main/kotlin")
        getByName("test").java.srcDirs("src/test/kotlin")
        getByName("androidTest").java.srcDirs("src/androidTest/kotlin")
        // The cross-language fixtures live with the Python tests that generate
        // them (tests/data). Host unit tests read them via a system property;
        // instrumented tests get them packaged as androidTest assets. One file,
        // both platforms — never two drifting copies.
        getByName("androidTest").assets.srcDirs("${rootDir.parentFile}/tests/data")
    }

    testOptions {
        unitTests.all { it.systemProperty("argus.fixtures", "${rootDir.parentFile}/tests/data") }
    }

    packaging {
        jniLibs {
            // Extract native libraries to disk at install time.
            //
            // Since API 23 the platform default is to leave uncompressed .so
            // files inside the APK and map them from there, which is why this
            // has to be asked for explicitly. QNN does not work that way: the
            // execution provider takes a filesystem `backend_path` and dlopens
            // libQnnHtp.so, which in turn loads the per-Hexagon skel onto the
            // DSP. With the libraries unextracted there is no such path, the EP
            // silently fails to register, every node falls back to CPU, and the
            // only reason that surfaces at all is that CPU fallback is
            // explicitly disabled. Found by QnnSessionTest on a real S25 Ultra.
            useLegacyPackaging = true
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity-ktx:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")

    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // ONNX Runtime with the QNN execution provider compiled in. This pulls
    // com.qualcomm.qti:qnn-runtime transitively, which is where libQnnHtp.so,
    // libQnnSystem.so and the per-Hexagon skels come from -- including
    // libQnnHtpV79Skel.so, the Snapdragon 8 Elite's. Both are on Maven Central;
    // no Qualcomm account or QAIRT SDK download is involved.
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.28.0")

    // WebSocket client for the ingest protocol (docs/PROTOCOL.md).
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    testImplementation("org.json:json:20240303")
}
